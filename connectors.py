"""
connectors.py - Ingestion Layer for the Strategic Market Intelligence Pipeline.

Architecture: Connector Pattern (Abstract Base Class)
──────────────────────────────────────────────────────
Every data source - whether a free RSS feed or a £500k/year LexisNexis
subscription - implements the same ``DataConnector`` interface.  This design
guarantees:

  * Plug-and-play extensibility: add Factiva, Bloomberg, Dow Jones, or any
    proprietary API by implementing one class.
  * Uniform data contract consumed by the Bronze layer and processor.
  * Isolation: authentication, pagination, and retry logic are fully
    encapsulated inside each connector.

Class hierarchy:
    DataConnector (ABC)
    ├── RSSConnector           - fully functional, zero credentials required
    └── LexisNexisConnector    - enterprise stub (OAuth2, pagination, filtering)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

import feedparser
import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contract - the normalised unit of ingestion
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RawArticle:
    """
    Immutable, normalised record produced by every connector.

    Downstream components (Bronze writer, processor) depend only on this
    contract - never on connector-specific structures.
    """

    source_name: str
    raw_text: str
    source_url: Optional[str] = None
    title: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_name:
            raise ValueError("RawArticle.source_name must not be empty.")
        if not self.raw_text or not self.raw_text.strip():
            raise ValueError("RawArticle.raw_text must not be empty.")


# ---------------------------------------------------------------------------
# Abstract Base Class - the connector interface contract
# ---------------------------------------------------------------------------
class DataConnector(ABC):
    """
    Abstract base class for all Market Intelligence data connectors.

    Subclasses MUST implement:
        - ``authenticate()``  - set up credentials / sessions
        - ``fetch()``         - yield ``RawArticle`` objects

    Subclasses MAY override:
        - ``close()``         - teardown (close HTTP sessions, etc.)
        - ``SOURCE_NAME``     - a human-readable identifier string
    """

    SOURCE_NAME: str = "unknown"

    def __init__(self) -> None:
        self._authenticated: bool = False
        logger.debug("Connector initialised: %s", self.__class__.__name__)

    @abstractmethod
    def authenticate(self) -> None:
        """
        Establish authentication with the data source.

        Implementations should handle:
        - OAuth2 token acquisition (LexisNexis, Factiva)
        - API key header injection (Bloomberg, Refinitiv)
        - Certificate-based mTLS (enterprise on-premise sources)

        Raises:
            AuthenticationError: if authentication fails after retries.
        """
        ...

    @abstractmethod
    def fetch(self, query: Optional[str] = None, **kwargs: Any) -> Iterator[RawArticle]:
        """
        Yield normalised ``RawArticle`` objects from the data source.

        Args:
            query: Optional keyword/boolean query string.
            **kwargs: Connector-specific parameters (date ranges, source
                      filters, pagination cursors, etc.)

        Yields:
            RawArticle: Normalised article ready for Bronze ingestion.
        """
        ...

    def close(self) -> None:
        """Release resources.  Override in subclasses as needed."""
        logger.debug("Connector closed: %s", self.__class__.__name__)

    # Context-manager support
    def __enter__(self) -> "DataConnector":
        self.authenticate()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source={self.SOURCE_NAME!r}, authenticated={self._authenticated})"


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class AuthenticationError(RuntimeError):
    """Raised when a connector cannot authenticate with its data source."""


class FetchError(RuntimeError):
    """Raised when a connector fails to retrieve data after exhausting retries."""


# ============================================================
#  CONNECTOR 1 - RSS / Atom (fully functional, zero cost)
# ============================================================
class RSSConnector(DataConnector):
    """
    Production-grade RSS/Atom connector using ``feedparser``.

    Designed as the default out-of-the-box connector so the pipeline
    is runnable with zero API credentials.  Feed URLs are configured
    in ``config.settings.rss_feed_urls``.

    Features:
      - ETag / Last-Modified cache headers to avoid re-downloading unchanged feeds
      - Per-entry deduplication via URL
      - Graceful handling of malformed feeds
      - Configurable max entries per feed
    """

    SOURCE_NAME: str = "rss"

    def __init__(
        self,
        feed_urls: Optional[List[str]] = None,
        max_entries_per_feed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._feed_urls: List[str] = feed_urls or settings.rss_feed_urls
        self._max_entries: int = max_entries_per_feed or settings.rss_max_entries_per_feed
        self._seen_urls: set[str] = set()
        # ETag / Modified cache: url → {"etag": ..., "modified": ...}
        self._feed_cache: Dict[str, Dict[str, str]] = {}

    def authenticate(self) -> None:
        """RSS feeds require no authentication."""
        self._authenticated = True
        logger.info("RSSConnector: no authentication required - %d feeds configured.", len(self._feed_urls))

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _parse_feed(self, url: str) -> feedparser.FeedParserDict:
        """Parse a single feed URL with retry logic and cache-control headers."""
        cache = self._feed_cache.get(url, {})
        parsed = feedparser.parse(
            url,
            etag=cache.get("etag"),
            modified=cache.get("modified"),
            agent="MarketIntelligencePipeline/1.0 (+https://github.com/your-org/market-intelligence)",
        )
        # Update cache headers for next call
        if parsed.get("etag"):
            self._feed_cache.setdefault(url, {})["etag"] = parsed.etag
        if parsed.get("modified"):
            self._feed_cache.setdefault(url, {})["modified"] = parsed.modified
        return parsed

    def fetch(self, query: Optional[str] = None, **kwargs: Any) -> Iterator[RawArticle]:
        """
        Yield articles from all configured RSS/Atom feeds.

        Args:
            query: Optional keyword filter applied to title + summary.
                   Case-insensitive substring match.
        """
        if not self._authenticated:
            self.authenticate()

        total_yielded = 0
        for feed_url in self._feed_urls:
            logger.info("RSSConnector: fetching feed → %s", feed_url)
            try:
                parsed = self._parse_feed(feed_url)
            except Exception as exc:
                logger.error("RSSConnector: failed to parse %s - %s", feed_url, exc)
                continue

            if parsed.status == 304:
                logger.info("RSSConnector: feed not modified (304) - %s", feed_url)
                continue

            feed_title = getattr(parsed.feed, "title", feed_url)
            entries = parsed.entries[: self._max_entries]

            for entry in entries:
                url = getattr(entry, "link", None)
                title: str = getattr(entry, "title", "Untitled")
                summary: str = getattr(entry, "summary", "") or ""
                content_list = getattr(entry, "content", [])
                full_content = content_list[0].value if content_list else ""
                raw_text = full_content or summary or title

                # Deduplication
                dedup_key = url or title
                if dedup_key in self._seen_urls:
                    logger.debug("RSSConnector: duplicate skipped - %s", dedup_key)
                    continue
                self._seen_urls.add(dedup_key)

                # Optional keyword filter
                if query:
                    haystack = f"{title} {summary}".lower()
                    if query.lower() not in haystack:
                        continue

                # Build metadata blob
                published = getattr(entry, "published", None)
                tags = [t.get("term") for t in getattr(entry, "tags", []) if t.get("term")]
                metadata: Dict[str, Any] = {
                    "feed_title": feed_title,
                    "feed_url": feed_url,
                    "published": published,
                    "tags": tags,
                    "authors": [a.get("name") for a in getattr(entry, "authors", []) if a.get("name")],
                }

                try:
                    article = RawArticle(
                        source_name=f"rss::{feed_title}",
                        raw_text=raw_text.strip(),
                        source_url=url,
                        title=title,
                        raw_metadata=metadata,
                    )
                    total_yielded += 1
                    yield article
                except ValueError as ve:
                    logger.warning("RSSConnector: skipping malformed entry - %s", ve)

            # Be a good citizen - small delay between feeds
            time.sleep(0.5)

        logger.info("RSSConnector: fetch complete - %d articles yielded.", total_yielded)


# ============================================================
#  CONNECTOR 2 - LexisNexis (Enterprise Stub)
# ============================================================
class LexisNexisConnector(DataConnector):
    """
    Enterprise stub for the LexisNexis News & Business Intelligence API.

    ─── Status: STUB - requires paid LexisNexis subscription ───

    This class demonstrates how the full connector would be implemented
    in a production environment.  It shows:

      1. OAuth2 Client Credentials flow (token acquisition + refresh)
      2. Cursor-based pagination for large result sets
      3. Boolean/proximity query syntax
      4. Rate-limit detection and back-off
      5. Schema-compliant ``RawArticle`` output

    To activate:
        1. Set LEXISNEXIS_CLIENT_ID and LEXISNEXIS_CLIENT_SECRET in .env
        2. Remove the ``raise NotImplementedError`` from authenticate()
           and uncomment the live implementation blocks.
        3. Register this connector in pipeline.py instead of RSSConnector.
    """

    SOURCE_NAME: str = "lexisnexis"

    # LexisNexis REST API endpoints
    _TOKEN_ENDPOINT: str = "/oauth/accesstoken"
    _SEARCH_ENDPOINT: str = "/news/v1/content/search"

    def __init__(self) -> None:
        super().__init__()
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._http: Optional[httpx.Client] = None

    # ------------------------------------------------------------------
    # Authentication - OAuth2 Client Credentials
    # ------------------------------------------------------------------
    def authenticate(self) -> None:
        """
        Acquire an OAuth2 Bearer token using the Client Credentials flow.

        In production this would POST to the LexisNexis token endpoint with
        ``grant_type=client_credentials``, parse the JSON response, and
        store the token along with its expiry time for automatic refresh.

        Raises:
            AuthenticationError: if credentials are missing or invalid.
        """
        if not settings.lexisnexis_client_id or not settings.lexisnexis_client_secret:
            raise AuthenticationError(
                "LexisNexis credentials not configured. "
                "Set LEXISNEXIS_CLIENT_ID and LEXISNEXIS_CLIENT_SECRET in your .env file."
            )

        logger.info("LexisNexisConnector: initiating OAuth2 Client Credentials flow …")

        # ---- STUB: replace with live implementation ----
        raise NotImplementedError(
            "LexisNexisConnector is a stub. "
            "Implement the OAuth2 token exchange below and remove this line."
        )

        # ---- LIVE implementation (commented for reference) ----
        # token_url = f"{settings.lexisnexis_base_url}{self._TOKEN_ENDPOINT}"
        # response = httpx.post(
        #     token_url,
        #     data={
        #         "grant_type": "client_credentials",
        #         "client_id": settings.lexisnexis_client_id,
        #         "client_secret": settings.lexisnexis_client_secret,
        #         "scope": settings.lexisnexis_scopes,
        #     },
        #     timeout=30,
        # )
        # response.raise_for_status()
        # payload = response.json()
        # self._access_token = payload["access_token"]
        # self._token_expires_at = time.time() + payload.get("expires_in", 3600) - 60
        # self._http = httpx.Client(
        #     base_url=settings.lexisnexis_base_url,
        #     headers={"Authorization": f"Bearer {self._access_token}"},
        #     timeout=60,
        # )
        # self._authenticated = True
        # logger.info("LexisNexisConnector: authenticated successfully.")

    def _refresh_token_if_needed(self) -> None:
        """Re-authenticate transparently if the Bearer token is expiring."""
        if time.time() >= self._token_expires_at:
            logger.info("LexisNexisConnector: access token expired - refreshing …")
            self.authenticate()

    # ------------------------------------------------------------------
    # Fetch - cursor-based pagination
    # ------------------------------------------------------------------
    def fetch(
        self,
        query: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sources: Optional[List[str]] = None,
        max_results: int = 200,
        **kwargs: Any,
    ) -> Iterator[RawArticle]:
        """
        Yield articles from LexisNexis using cursor-based pagination.

        Args:
            query:      Boolean/proximity query (e.g. ``"reinsurance AND cat bond"``).
            date_from:  ISO-8601 date string (``"2024-01-01"``).
            date_to:    ISO-8601 date string (``"2024-12-31"``).
            sources:    Optional list of LexisNexis source IDs to restrict search.
            max_results: Hard cap on total articles retrieved per call.

        Yields:
            RawArticle: Normalised article.

        Note:
            This method is a STUB.  The pagination loop and response parsing
            are shown here for architectural documentation purposes.
        """
        if not self._authenticated:
            self.authenticate()  # Will raise NotImplementedError in stub mode

        logger.info(
            "LexisNexisConnector: fetching - query=%r  date_from=%s  date_to=%s",
            query, date_from, date_to,
        )

        # ---- STUB: cursor-based pagination pattern ----
        # cursor: Optional[str] = None
        # fetched = 0
        #
        # while fetched < max_results:
        #     self._refresh_token_if_needed()
        #     params = {
        #         "$search": query or "reinsurance",
        #         "$expand": "Document",
        #         "$top": min(50, max_results - fetched),
        #     }
        #     if date_from:
        #         params["$filter"] = f"Date ge {date_from}"
        #     if date_to:
        #         params["$filter"] = params.get("$filter", "") + f" and Date le {date_to}"
        #     if cursor:
        #         params["$skiptoken"] = cursor
        #     if sources:
        #         params["$filter"] = params.get("$filter", "") + (
        #             " and Source/Id in (" + ",".join(sources) + ")"
        #         )
        #
        #     response = self._http.get(self._SEARCH_ENDPOINT, params=params)
        #
        #     if response.status_code == 429:
        #         retry_after = int(response.headers.get("Retry-After", 60))
        #         logger.warning("LexisNexis rate-limited - sleeping %ds", retry_after)
        #         time.sleep(retry_after)
        #         continue
        #
        #     response.raise_for_status()
        #     data = response.json()
        #
        #     for doc in data.get("value", []):
        #         content = doc.get("Document", {}).get("Content", "")
        #         yield RawArticle(
        #             source_name=self.SOURCE_NAME,
        #             raw_text=content,
        #             source_url=doc.get("ResultUrl"),
        #             title=doc.get("Title"),
        #             raw_metadata={
        #                 "source_id": doc.get("Source", {}).get("Id"),
        #                 "source_name": doc.get("Source", {}).get("Name"),
        #                 "published_date": doc.get("Date"),
        #                 "document_type": doc.get("DocumentType"),
        #             },
        #         )
        #         fetched += 1
        #
        #     cursor = data.get("@odata.nextLink")
        #     if not cursor or fetched >= max_results:
        #         break
        #
        # logger.info("LexisNexisConnector: fetch complete - %d articles yielded.", fetched)

        raise NotImplementedError(
            "LexisNexisConnector.fetch() is a stub. "
            "Uncomment the pagination block above after authentication is live."
        )

    def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._http:
            self._http.close()
            logger.debug("LexisNexisConnector: HTTP session closed.")
        super().close()


# ============================================================
#  CONNECTOR REGISTRY - add new connectors here
# ============================================================
CONNECTOR_REGISTRY: Dict[str, type[DataConnector]] = {
    "rss": RSSConnector,
    "lexisnexis": LexisNexisConnector,
    # "factiva": FactivaConnector,        # → implement similarly
    # "bloomberg": BloombergConnector,    # → implement similarly
    # "refinitiv": RefinitivConnector,    # → implement similarly
}


def get_connector(name: str, **kwargs: Any) -> DataConnector:
    """
    Factory function - resolves a connector by name from the registry.

    Args:
        name:    Key from ``CONNECTOR_REGISTRY`` (e.g. ``"rss"``).
        **kwargs: Passed to the connector's ``__init__``.

    Returns:
        Instantiated (but not yet authenticated) ``DataConnector``.

    Raises:
        KeyError: if ``name`` is not in the registry.
    """
    cls = CONNECTOR_REGISTRY.get(name.lower())
    if cls is None:
        available = list(CONNECTOR_REGISTRY.keys())
        raise KeyError(f"Connector '{name}' not found. Available: {available}")
    instance = cls(**kwargs)
    logger.debug("get_connector: created %r", instance)
    return instance

"""
Real browser automation tools powered by Playwright.

The browser context is module-level, so cookies and session state persist
across multiple tool calls within the same process. This is intentional —
it allows multi-step workflows (navigate → fill → click → read result).

Playwright is imported lazily, so this module can always be imported even
if playwright is not installed. Tools will return a helpful error message
if playwright is missing.

To install:
    pip install playwright
    playwright install chromium
"""
import logging
import re

from langchain_core.tools import tool

from observability import METRICS

logger = logging.getLogger(__name__)

# Module-level browser state — persists across tool calls for session continuity
_playwright_instance = None
_browser = None
_page = None

_MAX_PAGE_CHARS = 8000
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _ensure_browser():
    """Lazily initialise Playwright. Raises RuntimeError if not installed."""
    global _playwright_instance, _browser, _page

    if _playwright_instance is None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise RuntimeError(
                "Playwright is not installed. "
                "Run: pip install playwright && playwright install chromium"
            )
        _playwright_instance = sync_playwright().start()
        _browser = _playwright_instance.chromium.launch(headless=True)

    if _page is None or _page.is_closed():
        context = _browser.new_context(user_agent=_USER_AGENT)
        _page = context.new_page()

    return _page


def _clean_text(text: str) -> str:
    """Collapse excess whitespace and cap to _MAX_PAGE_CHARS."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()[:_MAX_PAGE_CHARS]


@tool(parse_docstring=True)
def browse_web(url: str) -> str:
    """
    Navigate to a URL using a real browser that executes JavaScript and returns
    the rendered page text. Use this for modern web apps that require JavaScript,
    or when you need to maintain session state (cookies) across multiple pages.
    For simple static pages, scrape_web is faster.

    Args:
        url: The full URL to navigate to (must include http:// or https://).

    Returns:
        string: The page title and visible text content (up to 8000 chars).
    """
    METRICS.increment("tool.browse_web")
    try:
        page = _ensure_browser()
        page.goto(url, wait_until="networkidle", timeout=20000)
        title = page.title()
        text = page.evaluate("() => document.body ? document.body.innerText : ''")
        clean = _clean_text(text)
        logger.info("browse_web: %s — %d chars returned", url, len(clean))
        return f"Page: {title}\nURL: {url}\n\n{clean}"
    except Exception as e:
        METRICS.record_error("browse_web", e)
        logger.warning("browse_web error for %s: %s", url, e)
        return f"Error browsing {url}: {e}"


@tool(parse_docstring=True)
def browser_click(selector: str) -> str:
    """
    Click an element on the currently loaded browser page.
    Call browse_web first to navigate to the target page.

    Args:
        selector: A CSS selector (e.g. '#submit-btn', 'button.login') or
                  the visible text of the element (e.g. 'Sign In', 'Submit').

    Returns:
        string: The visible page text after the click and any resulting navigation.
    """
    METRICS.increment("tool.browser_click")
    try:
        page = _ensure_browser()
        try:
            page.click(selector, timeout=5000)
        except Exception:
            page.get_by_text(selector, exact=False).first.click(timeout=5000)
        page.wait_for_load_state("networkidle", timeout=10000)
        text = page.evaluate("() => document.body ? document.body.innerText : ''")
        return f"Clicked '{selector}'. Page now shows:\n{_clean_text(text)}"
    except Exception as e:
        METRICS.record_error("browser_click", e)
        return f"Error clicking '{selector}': {e}"


@tool(parse_docstring=True)
def browser_fill(selector: str, value: str) -> str:
    """
    Type a value into a form field on the currently loaded browser page.
    Call browse_web first to navigate to the page, fill fields with this tool,
    then use browser_click to submit the form.

    Args:
        selector: CSS selector for the input field
                  (e.g. 'input[name="email"]', '#password', 'textarea').
        value: The text to enter into the field.

    Returns:
        string: Confirmation that the field was filled.
    """
    METRICS.increment("tool.browser_fill")
    try:
        page = _ensure_browser()
        page.fill(selector, value, timeout=5000)
        return f"Filled '{selector}'."
    except Exception as e:
        METRICS.record_error("browser_fill", e)
        return f"Error filling '{selector}': {e}"


def get_browser_tools_description() -> dict:
    """Return the browser tools registry for use in get_conversation_tools_description."""
    return {
        "browse_web": (
            browse_web,
            "Navigate to a URL with a real browser that handles JavaScript and maintains "
            "session cookies. Use for modern sites, SPAs, or multi-step browser workflows.",
        ),
        "browser_click": (
            browser_click,
            "Click a button or link on the current browser page by CSS selector or visible text. "
            "Call browse_web first to navigate.",
        ),
        "browser_fill": (
            browser_fill,
            "Fill a form input on the current browser page. "
            "Use browse_web to navigate, browser_fill to enter data, browser_click to submit.",
        ),
    }

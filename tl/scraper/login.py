"""`tl login`: run LOCALLY (not in Docker). Opens a visible Chromium window on X's login page,
you sign in yourself, and the resulting cookies/localStorage (Playwright "storage state")
are written to a file you then copy to the server's secrets volume.

This code never sees, asks for, or stores your password.
"""

from __future__ import annotations

import os
from pathlib import Path

from tl.scraper import selectors as S


def interactive_login(output: Path) -> Path:
    from playwright.sync_api import sync_playwright

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1100, "height": 900})
        page = context.new_page()
        page.goto(S.LOGIN_URL)
        print(
            "\nA Chromium window is open on X's login page.\n"
            "  1. Log in to the BOT account yourself (use a throwaway account).\n"
            "  2. Wait until the home timeline is visible.\n"
            "  3. Come back here and press Enter.\n"
        )
        input("Press Enter once the home timeline is showing... ")
        page.goto(S.HOME_URL, wait_until="domcontentloaded")
        try:
            page.wait_for_selector(S.TWEET_ARTICLE, timeout=60_000)
            print("Timeline detected.")
        except Exception:
            print("WARNING: no tweets detected on the home page; the saved session may not be logged in.")
        # Write with owner-only permissions.
        tmp = output.with_suffix(".tmp")
        context.storage_state(path=str(tmp))
        os.chmod(tmp, 0o600)
        tmp.replace(output)
        browser.close()
    return output

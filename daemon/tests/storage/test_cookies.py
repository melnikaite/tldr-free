"""Tests for src.storage.cookies — Cookie -> runtime artifact conversion.

Covers both public helpers (build_requests_session, write_netscape_cookie_file)
and the branches in the private _netscape_line formatter: HttpOnly handling,
wildcard-domain flag, secure flag, session vs explicit expiry, default path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.schemas import Cookie
from src.storage import cookies

# ---------------------------------------------------------------------------
# build_requests_session
# ---------------------------------------------------------------------------


def test_build_requests_session_empty_list() -> None:
    session = cookies.build_requests_session([])
    assert len(session.cookies) == 0


def test_build_requests_session_basic_cookie_loaded() -> None:
    c = Cookie(name="SID", value="abc", domain="youtube.com")
    session = cookies.build_requests_session([c])

    # Cookie is retrievable by name and carries its value.
    assert session.cookies.get("SID", domain="youtube.com") == "abc"


def test_build_requests_session_defaults_path_to_root() -> None:
    # path="" should be coerced to "/".
    c = Cookie(name="A", value="1", domain="example.com", path="")
    session = cookies.build_requests_session([c])
    jar = list(session.cookies)
    assert len(jar) == 1
    assert jar[0].path == "/"


def test_build_requests_session_secure_and_expiry() -> None:
    c = Cookie(
        name="S",
        value="v",
        domain="example.com",
        secure=True,
        expires=1_700_000_000.0,
    )
    session = cookies.build_requests_session([c])
    cookie = next(iter(session.cookies))
    assert cookie.secure is True
    # float expiry is coerced to int.
    assert cookie.expires == 1_700_000_000


def test_build_requests_session_none_expiry_is_session_cookie() -> None:
    c = Cookie(name="X", value="y", domain="example.com", expires=None)
    session = cookies.build_requests_session([c])
    cookie = next(iter(session.cookies))
    assert cookie.expires is None


def test_build_requests_session_http_only_sets_rest_attr() -> None:
    c = Cookie(name="H", value="v", domain="example.com", http_only=True)
    session = cookies.build_requests_session([c])
    cookie = next(iter(session.cookies))
    # HttpOnly lives in the non-standard "rest" bag for cookielib.
    assert cookie.has_nonstandard_attr("HttpOnly")


def test_build_requests_session_not_http_only_has_no_rest_attr() -> None:
    c = Cookie(name="H", value="v", domain="example.com", http_only=False)
    session = cookies.build_requests_session([c])
    cookie = next(iter(session.cookies))
    # rest=None is passed for non-HttpOnly cookies, so the "rest" bag is empty.
    assert not cookie._rest


def test_build_requests_session_multiple_cookies() -> None:
    cs = [
        Cookie(name="A", value="1", domain="a.com"),
        Cookie(name="B", value="2", domain="b.com"),
    ]
    session = cookies.build_requests_session(cs)
    assert session.cookies.get("A", domain="a.com") == "1"
    assert session.cookies.get("B", domain="b.com") == "2"


# ---------------------------------------------------------------------------
# write_netscape_cookie_file
# ---------------------------------------------------------------------------


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_write_netscape_file_creates_file_with_header(tmp_path: Path) -> None:
    c = Cookie(name="N", value="v", domain="example.com")
    path = cookies.write_netscape_cookie_file([c], tmp_path)

    assert path.exists()
    assert path.parent == tmp_path
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Netscape HTTP Cookie File")


def test_write_netscape_file_creates_missing_dir(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested"
    assert not nested.exists()
    path = cookies.write_netscape_cookie_file([], nested)
    assert nested.is_dir()
    assert path.exists()


def test_write_netscape_file_empty_cookies_only_header(tmp_path: Path) -> None:
    path = cookies.write_netscape_cookie_file([], tmp_path)
    # Header (3 lines incl trailing blank) and no cookie lines.
    text = path.read_text(encoding="utf-8")
    assert "# Netscape HTTP Cookie File" in text
    # No tab-separated data lines beyond the header.
    data_lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert data_lines == []


def test_write_netscape_file_basic_line_fields(tmp_path: Path) -> None:
    c = Cookie(
        name="SID",
        value="token",
        domain="youtube.com",
        path="/watch",
        secure=True,
        expires=1_700_000_000.0,
    )
    path = cookies.write_netscape_cookie_file([c], tmp_path)
    data = [ln for ln in _read_lines(path) if ln and not ln.startswith("#")]
    assert len(data) == 1
    fields = data[0].split("\t")
    # domain, include_subdomains, path, secure, expires, name, value
    assert fields == [
        "youtube.com",
        "FALSE",  # domain has no leading dot
        "/watch",
        "TRUE",  # secure
        "1700000000",
        "SID",
        "token",
    ]


def test_write_netscape_file_wildcard_domain_flag(tmp_path: Path) -> None:
    c = Cookie(name="W", value="v", domain=".example.com")
    path = cookies.write_netscape_cookie_file([c], tmp_path)
    data = [ln for ln in _read_lines(path) if ln and not ln.startswith("#HttpOnly")]
    data = [ln for ln in data if ln and not ln.startswith("# ")]
    line = data[0]
    fields = line.split("\t")
    assert fields[0] == ".example.com"
    assert fields[1] == "TRUE"  # leading dot -> include subdomains


def test_write_netscape_file_session_cookie_zero_expiry(tmp_path: Path) -> None:
    c = Cookie(name="Z", value="v", domain="example.com", expires=None)
    path = cookies.write_netscape_cookie_file([c], tmp_path)
    data = [ln for ln in _read_lines(path) if ln and not ln.startswith("#")]
    fields = data[0].split("\t")
    assert fields[4] == "0"  # None expiry -> 0 (session cookie)


def test_write_netscape_file_default_path(tmp_path: Path) -> None:
    c = Cookie(name="P", value="v", domain="example.com", path="")
    path = cookies.write_netscape_cookie_file([c], tmp_path)
    data = [ln for ln in _read_lines(path) if ln and not ln.startswith("#")]
    fields = data[0].split("\t")
    assert fields[2] == "/"


def test_write_netscape_file_http_only_domain_prefix(tmp_path: Path) -> None:
    c = Cookie(name="H", value="v", domain="example.com", http_only=True)
    path = cookies.write_netscape_cookie_file([c], tmp_path)
    text = path.read_text(encoding="utf-8")
    # HttpOnly cookies must use the "#HttpOnly_" domain prefix.
    assert "#HttpOnly_example.com" in text
    http_only_line = next(
        ln for ln in text.splitlines() if ln.startswith("#HttpOnly_")
    )
    fields = http_only_line.split("\t")
    assert fields[5] == "H"
    assert fields[6] == "v"


def test_write_netscape_file_not_secure_flag(tmp_path: Path) -> None:
    c = Cookie(name="N", value="v", domain="example.com", secure=False)
    path = cookies.write_netscape_cookie_file([c], tmp_path)
    data = [ln for ln in _read_lines(path) if ln and not ln.startswith("#")]
    fields = data[0].split("\t")
    assert fields[3] == "FALSE"


def test_write_netscape_file_multiple_cookies_one_line_each(tmp_path: Path) -> None:
    cs = [
        Cookie(name="A", value="1", domain="a.com"),
        Cookie(name="B", value="2", domain="b.com", http_only=True),
        Cookie(name="C", value="3", domain=".c.com"),
    ]
    path = cookies.write_netscape_cookie_file(cs, tmp_path)
    data = [
        ln
        for ln in _read_lines(path)
        if ln and (not ln.startswith("#") or ln.startswith("#HttpOnly_"))
    ]
    assert len(data) == 3


def test_write_netscape_file_returns_unique_paths(tmp_path: Path) -> None:
    # mkstemp gives a fresh file each call; two writes must not collide.
    p1 = cookies.write_netscape_cookie_file([], tmp_path)
    p2 = cookies.write_netscape_cookie_file([], tmp_path)
    assert p1 != p2
    assert p1.exists() and p2.exists()


def test_write_netscape_file_cleans_up_on_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If formatting a line raises, the temp file must be unlinked, not leaked."""
    created: list[Path] = []
    real_mkstemp = cookies.tempfile.mkstemp

    def tracking_mkstemp(*args: object, **kwargs: object):
        fd, path_str = real_mkstemp(*args, **kwargs)
        created.append(Path(path_str))
        return fd, path_str

    monkeypatch.setattr(cookies.tempfile, "mkstemp", tracking_mkstemp)

    def boom(_cookie: Cookie) -> str:
        raise RuntimeError("formatting failed")

    monkeypatch.setattr(cookies, "_netscape_line", boom)

    c = Cookie(name="X", value="v", domain="example.com")
    with pytest.raises(RuntimeError, match="formatting failed"):
        cookies.write_netscape_cookie_file([c], tmp_path)

    # The partially-written file must have been removed.
    assert created, "mkstemp should have been called"
    assert not created[0].exists()

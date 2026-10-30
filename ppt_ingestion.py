"""Convert legacy PPT files in an isolated temporary LibreOffice profile."""

import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv


load_dotenv(Path(__file__).with_name(".env"))
PPT_CONVERSION_TIMEOUT_SECONDS = 120


def find_libreoffice() -> str:
    configured = os.getenv("LIBREOFFICE_PATH", "").strip()
    if configured:
        resolved = shutil.which(configured)
        if resolved:
            return resolved
        if Path(configured).is_file():
            return str(Path(configured).resolve())
        raise ValueError("LIBREOFFICE_PATH does not identify a LibreOffice executable.")

    for command in ("soffice", "libreoffice"):
        resolved = shutil.which(command)
        if resolved:
            return resolved

    candidates = []
    for directory in (
        os.getenv("PROGRAMFILES", "C:/Program Files"),
        os.getenv("PROGRAMFILES(X86)", "C:/Program Files (x86)"),
    ):
        candidates.extend(
            Path(directory) / "LibreOffice" / "program" / executable
            for executable in ("soffice.com", "soffice.exe")
        )
    candidates.append(Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    raise ValueError(
        "Legacy .ppt files require LibreOffice. Install LibreOffice, restart "
        "the app, and upload again. If necessary, set LIBREOFFICE_PATH in .env "
        "to its soffice.com or soffice executable."
    )


@contextmanager
def converted_pptx(source: Path):
    executable = find_libreoffice()
    with TemporaryDirectory(prefix="task6-ppt-") as directory:
        root = Path(directory)
        input_path = root / "input.ppt"
        output_directory = root / "converted"
        output_directory.mkdir()
        shutil.copyfile(source, input_path)
        profile = (root / "profile").resolve().as_uri()
        command = [
            executable,
            f"-env:UserInstallation={profile}",
            "--headless",
            "--convert-to", "pptx:Impress MS PowerPoint 2007 XML",
            "--outdir", str(output_directory),
            str(input_path),
        ]
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=PPT_CONVERSION_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ValueError("PPT conversion exceeded the 120-second limit.") from error
        except OSError as error:
            raise ValueError("LibreOffice could not be started for PPT conversion.") from error
        output = output_directory / "input.pptx"
        if process.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            raise ValueError(
                "LibreOffice could not convert this PPT file. Check that it "
                "opens correctly and is not password-protected."
            )
        yield output

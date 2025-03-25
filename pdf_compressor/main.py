from __future__ import annotations

import os
import re
import shutil
import tempfile
from argparse import ArgumentParser
from glob import glob
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict
from zipfile import ZipFile

from pdf_compressor.ilovepdf import Compress, ILovePDF
from pdf_compressor.utils import ROOT, del_or_keep_compressed, load_dotenv

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_SUFFIX = "-compressed"
API_KEY_KEY = "ILOVEPDF_PUBLIC_KEY"
MISSING_API_KEY_ERR = KeyError(
    "pdf-compressor needs an iLovePDF public key to access its API. Set one "
    "with pdf-compressor --set-api-key project_public_7af905e... or as environment "
    f"variable {API_KEY_KEY}"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Compress PDFs using iLovePDF's API."""
    parser = ArgumentParser(
        description="Batch compress PDFs on the command line. Powered by iLovePDF.com.",
        allow_abbrev=False,
    )

    parser.add_argument("filenames", nargs="*", help="List of PDF files to compress.")

    parser.add_argument(
        "--set-api-key",
        help="Set the public key needed to authenticate with the iLovePDF API. Exits "
        "immediately afterwards ignoring all other flags.",
    )

    parser.add_argument(
        "--password",
        type=str,
        default="",
        help="Password for protected PDF files. All files will use the same password. "
        "Protected PDFs with different passwords must be compressed one by one.",
    )
    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "-i",
        "--inplace",
        action="store_true",
        help="Whether to compress PDFs in place. Defaults to False.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="",
        help="Output directory for compressed PDFs. Defaults to the current working "
        "directory.",
    )

    group.add_argument(
        "-s",
        "--suffix",
        default=DEFAULT_SUFFIX,
        help="String to append to the filename of compressed PDFs. Mutually exclusive "
        "with --inplace flag.",
    )

    parser.add_argument(
        "--report-quota",
        action="store_true",
        help="Report how much of the monthly quota for the current API key has been "
        "used up.",
    )

    parser.add_argument(
        "--compression-level",
        "--cl",
        choices=("low", "recommended", "extreme"),
        default="recommended",
        help="How hard to squeeze the file size. 'extreme' noticeably degrades image "
        "quality. Defaults to 'recommended'.",
    )

    parser.add_argument(
        "--min-size-reduction",
        "--min-red",
        type=int,
        choices=range(101),
        metavar="[0-100]",  # prevents long list in argparse help message
        help="How much compressed files need to be smaller than originals (in percent) "
        "for them to be kept. Defaults to 10 when also passing -i/--inplace, else 0."
        "For example, when compressing files in-place and only achieving 5%% file size "
        "reduction, the compressed file will be discarded.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="When true, iLovePDF won't process the request but will output the "
        "parameters received by the server. Defaults to False.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="When true, progress will be reported while tasks are running. Also prints"
        " full file paths to compressed files instead of file name only. Defaults "
        "to False.",
    )

    parser.add_argument(
        "--on-no-files",
        choices=("error", "ignore"),
        default="ignore",
        help="What to do when no input PDFs received. One of 'ignore' or 'error', "
        "former exits 0, latter throws ValueError. Can be useful when using "
        "pdf-compressor in shell scripts. Defaults to 'ignore'.",
    )

    parser.add_argument(
        "--on-bad-files",
        choices=("error", "warn", "ignore"),
        default="error",
        help="How to behave when receiving input files that don't appear to be PDFs. "
        "One of 'error', 'warn', 'ignore'. Error will be TypeError. "
        "Defaults to 'error'.",
    )

    parser.add_argument(
        "--write-stats-path",
        type=str,
        default="",
        help="File path to write a CSV, Excel or other pandas supported file format "
        "with original vs compressed file sizes and actions taken on each input file",
    )

    pkg_version = version(pkg_name := "pdf-compressor")
    parser.add_argument(
        "-v", "--version", action="version", version=f"{pkg_name} v{pkg_version}"
    )
    args = parser.parse_args(argv)

    if new_key := args.set_api_key:
        if not new_key.startswith("project_public_"):
            raise ValueError(
                f"invalid API key, must start with 'project_public_', got {new_key=}"
            )

        env_path = Path(ROOT) / ".env"
        with open(env_path, "w+", encoding="utf8") as file:
            file.write(f"ILOVEPDF_PUBLIC_KEY={new_key}\n")

        return 0

    load_dotenv()
    try:
        api_key = os.environ[API_KEY_KEY]
        if not api_key:
            raise MISSING_API_KEY_ERR
    except KeyError:
        raise MISSING_API_KEY_ERR

    if args.report_quota:
        remaining_files = ILovePDF(api_key).get_quota()

        print(f"Remaining files in this billing cycle: {remaining_files:,}")

        return 0

    return compress(**vars(args))


def compress(
    filenames: Sequence[str],
    *,
    inplace: bool = False,
    outdir: str = "",
    suffix: str = DEFAULT_SUFFIX,
    compression_level: str = "recommended",
    min_size_reduction: int | None = None,
    debug: bool = False,
    verbose: bool = False,
    on_no_files: str = "ignore",
    on_bad_files: str = "error",
    write_stats_path: str = "",
    password: str = "",
    **kwargs: Any,  # noqa: ARG001
) -> int:
    """Compress PDFs using iLovePDF's API.

    Args:
        filenames (list[str]): List of PDF files to compress.
        inplace (bool): Whether to compress PDFs in place.
        outdir (str): Output directory for compressed PDFs. Defaults to the current
            working directory.
        suffix (str): String to append to the filename of compressed PDFs.
        compression_level (str): How hard to squeeze the file size.
        min_size_reduction (int): How much compressed files need to be smaller than
            originals (in percent) for them to be kept.
        debug (bool): When true, iLovePDF won't process the request but will output the
            parameters received by the server.
        verbose (bool): When true, progress will be reported while tasks are running.
        on_no_files (str): What to do when no input PDFs received.
        on_bad_files (str): How to behave when receiving input files that don't appear
            to be PDFs.
        write_stats_path (str): File path to write a CSV, Excel or other pandas
            supported file format with original vs compressed file sizes and actions
            taken on each input file
        password (str): Password to open PDFs in case they have one. Defaults to "".
            TODO There's currently no way of passing different passwords for different
            files. PDFs with different passwords must be compressed one by one.
        **kwargs: Additional keywords are ignored.

    Returns:
        int: 0 if successful, else error code.
    """
    if min_size_reduction is None:
        min_size_reduction = 10 if inplace else 0

    load_dotenv()
    try:
        api_key = os.environ[API_KEY_KEY]
        if not api_key:
            raise MISSING_API_KEY_ERR
    except KeyError:
        raise MISSING_API_KEY_ERR

    if not (inplace or suffix):
        raise ValueError(
            "Files must either be compressed in-place (--inplace) or you must specify a"
            " non-empty suffix to append to the name of compressed files."
        )

    # Convert input filenames to absolute paths using Path objects
    uniq_files = set()
    for fn in filenames:
        # Use Path to standardize path handling across OS
        path_obj = Path(fn.strip())
        # Convert to absolute path and normalize
        abs_path = str(path_obj.absolute())
        uniq_files.add(abs_path)
    
    # Sort the unique files for consistent processing order
    uniq_files_sorted = sorted(uniq_files)
    
    # for each directory received glob for all PDFs in it
    file_paths = []
    for file_path in uniq_files_sorted:
        if os.path.isdir(file_path):
            # Use Path for better cross-platform path handling
            search_path = Path(file_path) / "**" / "*.pdf*"
            # Convert search_path to string for glob
            glob_pattern = str(search_path)
            matches = glob(glob_pattern, recursive=True)
            file_paths.extend([str(Path(match).absolute()) for match in matches])
        else:
            file_paths.append(file_path)

    # match files case insensitively ending with .pdf(,a,x) and possible white space
    pdf_paths = []
    for f in file_paths:
        # Use Path object for better path handling
        path_obj = Path(f.rstrip())
        if re.match(r".*\.pdf[ax]?\s*$", path_obj.name.lower()):
            pdf_paths.append(str(path_obj))
    
    not_pdf_paths = set(file_paths) - set(pdf_paths)

    if on_bad_files == "error" and len(not_pdf_paths) > 0:
        raise ValueError(
            f"Input files must be PDFs, got {len(not_pdf_paths):,} files with "
            f"unexpected extension: {', '.join(not_pdf_paths)}"
        )
    if on_bad_files == "warn" and len(not_pdf_paths) > 0:
        print(
            f"Warning: Got {len(not_pdf_paths):,} input files without '.pdf' "
            f"extension: {', '.join(not_pdf_paths)}"
        )

    if verbose:
        if len(pdf_paths) > 0:
            print(f"PDFs to be compressed with iLovePDF: {len(pdf_paths):,}")
        else:
            print("Nothing to do: received no input PDF files.")

    if len(pdf_paths) == 0:
        if on_no_files == "error":
            raise ValueError("No input files provided")
        return 0

    task = Compress(
        api_key, compression_level=compression_level, debug=debug, password=password
    )
    task.verbose = verbose

    for pdf in pdf_paths:
        task.add_file(pdf)

    task.process()

    # Create a temporary directory to safely extract the downloaded file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download the file to the temporary directory
        downloaded_file = task.download(save_to_dir=temp_dir)
        task.delete_current_task()

        # If in debug mode, no need to process the files
        if debug:
            return 0

        # Handle the downloaded file (which may be a zip file)
        min_size_red = min_size_reduction or (10 if inplace else 0)
        
        # Process the downloaded file
        stats = process_compressed_files(
            pdf_paths,
            downloaded_file,
            outdir=outdir, 
            inplace=inplace,
            suffix=suffix,
            min_size_reduction=min_size_red,
            verbose=verbose,
        )

    if write_stats_path:
        try:
            import pandas as pd  # noqa: PLC0415
        except ImportError:
            err_msg = "To write stats to file, install pandas: pip install pandas"
            raise ImportError(err_msg) from None

        df_stats = pd.DataFrame(stats).T
        df_stats.index.name = "file"
        stats_path_lower = write_stats_path.strip().lower()

        # Ensure the stats path is normalized
        stats_file_path = Path(write_stats_path)

        if ".csv" in stats_path_lower:
            df_stats.to_csv(stats_file_path, float_format="%.4f")
        elif ".xlsx" in stats_path_lower or ".xls" in stats_path_lower:
            df_stats.to_excel(stats_file_path, float_format="%.4f")
        elif ".json" in stats_path_lower:
            df_stats.to_json(stats_file_path)
        elif ".html" in stats_path_lower:
            df_stats.to_html(stats_file_path, float_format="%.4f")

    return 0


def process_compressed_files(
    original_files: list[str],
    zip_file_path: str,
    outdir: str = "",
    inplace: bool = False,
    suffix: str = DEFAULT_SUFFIX,
    min_size_reduction: int = 0,
    verbose: bool = False,
) -> Dict[str, Dict]:
    """Process the compressed files from the zip file.
    
    Args:
        original_files (list[str]): List of original PDF file paths
        zip_file_path (str): Path to the downloaded ZIP file from iLovePDF
        outdir (str): Output directory for extracted files
        inplace (bool): Whether to replace original files
        suffix (str): Suffix to add to filenames if not replacing
        min_size_reduction (int): Minimum size reduction percentage to keep files
        verbose (bool): Whether to print verbose output
        
    Returns:
        Dict: Stats dictionary with file information
    """
    stats = {}
    
    # Create the output directory if specified and doesn't exist
    out_dir_path = Path(outdir) if outdir else Path.cwd()
    os.makedirs(out_dir_path, exist_ok=True)
    
    # Extract the files to a temporary directory
    with tempfile.TemporaryDirectory() as extract_dir:
        with ZipFile(zip_file_path, 'r') as zipf:
            # Extract all files to the temporary directory
            zipf.extractall(extract_dir)
            
            # Get a mapping of filenames in the ZIP to their paths in the extract_dir
            extracted_files = {}
            for file_info in zipf.infolist():
                if not file_info.is_dir():
                    # Extract only the filename without any directories
                    base_name = os.path.basename(file_info.filename)
                    # Map to the path in the extract_dir
                    extracted_files[base_name] = os.path.join(extract_dir, file_info.filename)
        
        # Process each original file
        for i, original_path in enumerate(original_files):
            # Get the base name of the original file
            original_file = Path(original_path)
            filename = original_file.name
            
            # Find the corresponding extracted file
            # First try exact match, then try with index prefix that iLovePDF might add
            extracted_path = None
            if filename in extracted_files:
                extracted_path = extracted_files[filename]
            else:
                # Try looking for files that might have been prefixed with an index
                for extracted_name, path in extracted_files.items():
                    # Check if filename is in the extracted name (might have prefix)
                    if filename in extracted_name:
                        extracted_path = path
                        break
            
            if not extracted_path:
                # If no match found, skip this file
                print(f"Warning: No matching compressed file found for {filename}")
                continue
            
            # Get file sizes
            original_size = os.path.getsize(original_path)
            compressed_size = os.path.getsize(extracted_path)
            size_diff = original_size - compressed_size
            size_reduction_pct = round(size_diff / original_size * 100)
            
            # Decide whether to keep the compressed file
            if size_reduction_pct < min_size_reduction:
                action = f"Discarded (only {size_reduction_pct}% smaller)"
                if verbose:
                    print(f"{i+1} '{filename}': {original_size/1024:.1f}KB -> "
                          f"{compressed_size/1024:.1f}KB which is "
                          f"{size_diff/1024:.1f}KB = {size_reduction_pct}% smaller. "
                          f"Compressed file discarded (< {min_size_reduction}% reduction).")
            else:
                # Determine the destination path
                if inplace:
                    dest_path = original_path
                    # Move the original file to trash/delete it
                    try:
                        # Try to use send2trash if available
                        from send2trash import send2trash
                        send2trash(original_path)
                        if verbose:
                            print(f"Old file moved to trash.")
                    except ImportError:
                        # If send2trash isn't available, just use os.remove
                        os.remove(original_path)
                        if verbose:
                            print(f"Old file deleted.")
                    action = "Replaced original"
                else:
                    # Add suffix to the filename
                    stem = original_file.stem
                    dest_path = os.path.join(out_dir_path, f"{stem}{suffix}{original_file.suffix}")
                    action = f"Saved with suffix '{suffix}'"
                
                # Copy the compressed file to the destination
                shutil.copy2(extracted_path, dest_path)
                
                if verbose:
                    print(f"{i+1} '{filename}': {original_size/1024:.1f}KB -> "
                          f"{compressed_size/1024:.1f}KB which is "
                          f"{size_diff/1024:.1f}KB = {size_reduction_pct}% smaller.")
            
            # Save stats
            stats[filename] = {
                "original size (B)": original_size,
                "compressed size (B)": compressed_size,
                "size reduction (B)": size_diff,
                "size reduction (%)": size_reduction_pct,
                "action": action,
            }
    
    return stats


if __name__ == "__main__":
    raise SystemExit(main())
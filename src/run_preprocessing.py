import argparse
import logging

from .utils import list_ims_files
from .preprocessing import fit_global_scaler, create_memmap_dataset
from .config import CONFIG, ensure_output_dirs, configure_logging


def main():
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process (demo)")
    parser.add_argument("--data-folder", type=str, default=None, help="Override data folder")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    configure_logging(logging.DEBUG if args.verbose else logging.INFO)
    ensure_output_dirs()

    logging.info("Discovering IMS files...")

    folder = args.data_folder or CONFIG["data_folder"]
    files = list_ims_files(folder, seq_length=CONFIG["sequence_length"])
    if args.limit:
        files = files[: args.limit]

    logging.info("%d usable files found.", len(files))

    logging.info("Fitting global scaler...")
    scaler = fit_global_scaler(files)

    logging.info("Creating memmap dataset...")
    create_memmap_dataset(files, scaler)

    logging.info("Preprocessing pipeline complete.")


if __name__ == "__main__":
    main()

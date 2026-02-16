from .utils import list_ims_files
from .preprocessing import fit_global_scaler, create_memmap_dataset
from .config import CONFIG


def main():

    print("Discovering IMS files...")

    files = list_ims_files(
        CONFIG["data_folder"],
        seq_length=CONFIG["sequence_length"]
    )

    print(f"{len(files)} usable files found.")

    print("\nFitting global scaler...")
    scaler = fit_global_scaler(files)

    print("\nCreating memmap dataset...")
    create_memmap_dataset(files, scaler)

    print("\n✅ Preprocessing pipeline complete.")


if __name__ == "__main__":
    main()

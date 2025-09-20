import sys
import gzip
import subprocess
import os

# --- Configuration ---
FULL_DBNSFP_PATH = "/home/hassan2/dbNSFP5.2a_grch38.gz"
CUSTOM_DB_PATH_DATA_ONLY = "/home/hassan2/genixrl_scores_db.data_only.tsv"
CUSTOM_DB_PATH_SORTED = "/home/hassan2/genixrl_scores_db.sorted.tsv"
CUSTOM_DB_FINAL_PATH = "/home/hassan2/genixrl_scores_db.tsv.gz"

COLUMNS_TO_KEEP = [
    '#chr', 'pos(1-based)', 'ref', 'alt', 'ClinPred_score',
    'BayesDel_addAF_score', 'BayesDel_noAF_score', 'MetaRNN_score'
]

def clean_metarnn_score(score_str):
    """Clean MetaRNN_score by selecting the first valid numeric value."""
    if score_str == '.' or not score_str:
        return '.'
    # Split on ';' or ',' and filter out empty or invalid entries
    scores = [s.strip() for s in score_str.replace(';', ',').split(',') if s.strip()]
    for score in scores:
        try:
            float_score = float(score)
            return str(float_score)
        except ValueError:
            continue
    return '.'  # Return '.' if no valid numeric score is found

def main():
    print(f"Starting to process the full dbNSFP file: {FULL_DBNSFP_PATH}")
    print("This will take a significant amount of time...")

    header_to_write = ""

    # Step 1: Extract data (NO HEADER) and find column indices
    with gzip.open(FULL_DBNSFP_PATH, 'rt') as f_in:
        header_line = f_in.readline().strip()
        all_columns = header_line.split('\t')
        
        try:
            indices_to_keep = [all_columns.index(col) for col in COLUMNS_TO_KEEP]
            metarnn_idx = all_columns.index('MetaRNN_score')  # Index for cleaning
        except ValueError as e:
            print(f"Error: A required column was not found in the dbNSFP header: {e}", file=sys.stderr)
            sys.exit(1)
            
        header_to_write = '\t'.join(COLUMNS_TO_KEEP)
        print("Found all required columns. Extracting data (this is the long part)...")
        
        with open(CUSTOM_DB_PATH_DATA_ONLY, 'w') as f_out:
            processed_lines = 0
            for line in f_in:
                fields = line.strip().split('\t')
                selected_data = [fields[i] for i in indices_to_keep]
                # Remove the '#' from the chromosome name in data lines
                selected_data[0] = selected_data[0].replace('#', '')
                # Clean MetaRNN_score
                selected_data[COLUMNS_TO_KEEP.index('MetaRNN_score')] = clean_metarnn_score(selected_data[COLUMNS_TO_KEEP.index('MetaRNN_score')])
                f_out.write('\t'.join(selected_data) + '\n')
                
                processed_lines += 1
                if processed_lines % 5000000 == 0:
                    print(f"  ...processed {processed_lines:,} variants...")
                    sys.stdout.flush()

    print(f"Extraction complete. Data-only file created at: {CUSTOM_DB_PATH_DATA_ONLY}")

    # Step 2: Sort the data-only file
    print("\nSorting the data file...")
    # Sort by chromosome (version sort 'V'), then by position (numeric sort 'n')
    sort_command = f"sort -k1,1V -k2,2n {CUSTOM_DB_PATH_DATA_ONLY} > {CUSTOM_DB_PATH_SORTED}"
    subprocess.run(sort_command, shell=True, check=True)
    print(f"Sorting complete. Sorted data file at: {CUSTOM_DB_PATH_SORTED}")
    
    # Step 3: Add header back, compress, and index
    print("\nAdding header, compressing with bgzip, and indexing...")
    
    # Combine header and sorted data, then compress
    final_command = (
        f'(echo "{header_to_write}" && cat {CUSTOM_DB_PATH_SORTED}) | '
        f'bgzip > {CUSTOM_DB_FINAL_PATH}'
    )
    subprocess.run(final_command, shell=True, executable='/bin/bash', check=True)

    print(f"Compression complete. Final database at: {CUSTOM_DB_FINAL_PATH}")

    # Index with Tabix
    tabix_command = f"tabix -p vcf -S 1 {CUSTOM_DB_FINAL_PATH}"
    subprocess.run(tabix_command, shell=True, check=True)
    
    print("\n Success! Your custom, indexed database is ready.")
    
    # Step 4: Clean up intermediate files 
    print("\nCleaning up intermediate files...")
    if os.path.exists(CUSTOM_DB_PATH_DATA_ONLY):
        os.remove(CUSTOM_DB_PATH_DATA_ONLY)
    if os.path.exists(CUSTOM_DB_PATH_SORTED):
        os.remove(CUSTOM_DB_PATH_SORTED)
    print("Cleanup complete.")

if __name__ == "__main__":
    main()
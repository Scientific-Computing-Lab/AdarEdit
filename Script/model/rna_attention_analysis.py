import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import RNA
import argparse
import os


def create_composite_figure(structure_attention_csv, output_path, window_size=50):
    df = pd.read_csv(structure_attention_csv)
    df = df[df['Base'] != 'N']
    df_filtered = df[(df['Position_relative_to_editing_site'].between(-window_size, window_size)) &
                     (df['Position_Side'] == 'same side')]

    def loop_size_group(size):
        if size == 1:
            return '1'
        elif size == 2:
            return '2'
        elif size == 3:
            return '3'
        elif 4 <= size <= 6:
            return '4-6'
        else:
            return None

    loop_df = df_filtered[(df_filtered['Loop_Size'].notna()) &
                          (df_filtered['Base'] != 'N') &
                          (df_filtered['Loop_Partner_Base'] != 'N')].copy()
    loop_df['Loop_Size_Group'] = loop_df['Loop_Size'].apply(loop_size_group)

    # Pastel theme globally
    sns.set(style='whitegrid', palette='pastel')

    # Composite figure 3x3
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    markersize = 4

    # Plot A: Mean Attention per Position
    mean_attention = df_filtered.groupby('Position_relative_to_editing_site')['Attention'].mean()
    axes[0, 0].plot(mean_attention.index, mean_attention.values, marker='o', markersize=markersize)
    axes[0, 0].axvline(x=0, color='red', linestyle='--')
    axes[0, 0].set_title('Mean Attention per Position')
    axes[0, 0].set_xlabel('')
    axes[0, 0].set_ylabel('Mean Attention')

    # Plot B: Mean Attention per Base
    df_base_filtered = df_filtered[df_filtered['Base'].isin(['A', 'C', 'G', 'U','T'])].copy()
    sns.lineplot(data=df_base_filtered, x='Position_relative_to_editing_site', y='Attention',
                 hue='Base', ax=axes[0, 1], marker='o', ci=None, markersize=markersize)
    axes[0, 1].axvline(x=0, color='red', linestyle='--')
    axes[0, 1].set_title('Mean Attention per Base')
    axes[0, 1].set_xlabel('')
    axes[0, 1].set_ylabel('Mean Attention')

    # Plot C: Mean Attention by Loop Status
    sns.lineplot(data=df_filtered, x='Position_relative_to_editing_site', y='Attention',
                 hue='Position_Type', ax=axes[0, 2], marker='o', ci=None, markersize=markersize)
    axes[0, 2].axvline(x=0, color='red', linestyle='--')
    axes[0, 2].set_title('Loop vs Paired Positions')
    axes[0, 2].set_xlabel('')
    axes[0, 2].set_ylabel('Mean Attention')

    # Plot D: Mean Attention by Editing Status
    sns.lineplot(data=df_filtered, x='Position_relative_to_editing_site', y='Attention',
                 hue='Edited', ax=axes[1, 0], marker='o', ci=None, markersize=markersize)
    axes[1, 0].axvline(x=0, color='red', linestyle='--')
    axes[1, 0].set_title('Edited vs Not Edited')
    axes[1, 0].set_xlabel('Position')
    axes[1, 0].set_ylabel('Mean Attention')

    # Plot E: Attention by Loop Size Group
    sns.lineplot(data=loop_df, x='Position_relative_to_editing_site', y='Attention',
                 hue='Loop_Size_Group', ax=axes[1, 1], marker='o', ci=None,
                 hue_order=['1', '2', '3', '4-6'], markersize=markersize)
    axes[1, 1].axvline(x=0, color='red', linestyle='--')
    axes[1, 1].set_title('Attention by Loop Size')
    axes[1, 1].set_xlabel('Position')
    axes[1, 1].set_ylabel('Mean Attention')

    # Plot F: Removed (was attention diff)
    axes[1, 2].axis('off')

    # Remaining: empty
    axes[2, 0].axis('off')
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')

    # Save figure
    fig.suptitle('RNA Editing Attention Analysis', fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(os.path.join(output_path, "attention_composite_figure.pdf"))
    fig.savefig(os.path.join(output_path, "attention_composite_figure.png"))

# Function to generate all plots
def generate_attention_plots(df, output_dir,window_size):
    df = df[df['Base'] != 'N']
    df_filtered = df[(df['Position_relative_to_editing_site'].between(-window_size, window_size)) &
                     (df['Position_Side'] == 'same side')]

    sns.set(style='whitegrid', palette='deep')

    # Plot 1: Mean Attention per Position
    plt.figure(figsize=(12, 7))
    mean_attention = df_filtered.groupby('Position_relative_to_editing_site')['Attention'].mean()
    sns.lineplot(x=mean_attention.index , y=mean_attention.values, marker='o')
    plt.axvline(x=0, color='red', linestyle='--', label='Editing Site')
    plt.title('Mean Attention per Position (-50 to +50)')
    plt.xlabel('Position Relative to Editing Site')
    plt.ylabel('Mean Attention')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_attention_per_position.png"))

    # Plot 2: Mean Attention by Loop Status
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df_filtered, x='Position_relative_to_editing_site', y='Attention',
                 hue='Position_Type', marker='o', ci=None)
    plt.axvline(x=0, color='red', linestyle='--', label='Editing Site')
    plt.title('Mean Attention per Position by Loop Status')
    plt.xlabel('Position Relative to Editing Site')
    plt.ylabel('Mean Attention')
    plt.legend(title='Position Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_attention_by_loop_status.png"))

    # Plot 3: Mean Attention per Base by Loop Status
    g = sns.FacetGrid(df_filtered, col="Position_Type", height=7, aspect=1.2)
    g.map_dataframe(sns.lineplot, x='Position_relative_to_editing_site', y='Attention', hue='Base', marker='o', ci=None)
    g.add_legend()
    g.set_axis_labels("Position Relative to Editing Site", "Mean Attention")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Mean Attention per Base by Loop Status')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_attention_per_base_loop_status.png"))

    # Plot 4: Loop Partner Base Attention
    loop_partner_df = df_filtered[(df_filtered['Position_Type'] == 'loop/unpaired') &
                                  (~df_filtered['Loop_Partner_Base'].isna()) &
                                  (df_filtered['Loop_Partner_Base'] != 'N')]
    pivot_partner = loop_partner_df.pivot_table(index='Position_relative_to_editing_site',
                                                columns='Loop_Partner_Base', values='Attention')
    pivot_partner.plot(figsize=(12, 7), marker='o')
    plt.axvline(x=0, color='red', linestyle='--', label='Editing Site')
    plt.title('Mean Attention per Loop Position by Loop Partner Base')
    plt.xlabel('Position Relative to Editing Site')
    plt.ylabel('Mean Attention')
    plt.legend(title='Loop Partner Base')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_attention_loop_partner_base.png"))

    # Plot 5: Mean Attention by Editing Status
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df_filtered, x='Position_relative_to_editing_site', y='Attention', hue='Edited', marker='o', ci=None)
    plt.axvline(x=0, color='red', linestyle='--', label='Editing Site')
    plt.title('Mean Attention by Editing Status')
    plt.xlabel('Position Relative to Editing Site')
    plt.ylabel('Mean Attention')
    plt.legend(title='Edited')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_attention_editing_status.png"))

    # Plot 6: Attention Difference (Edited - Not Edited) by Base
    base_diff = (df_filtered[df_filtered['Edited'] == 'yes'].groupby(['Position_relative_to_editing_site', 'Base'])['Attention'].mean() -
                 df_filtered[df_filtered['Edited'] == 'no'].groupby(['Position_relative_to_editing_site', 'Base'])['Attention'].mean()).unstack()
    base_diff.plot(figsize=(12, 7), marker='o')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title('Attention Difference (Edited - Not Edited) by Base')
    plt.xlabel('Position Relative to Editing Site')
    plt.ylabel('Attention Difference')
    plt.legend(title='Base')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attention_diff_by_base.png"))

    # Plot 7: Statistical significance by position (t-test)
    positions = range(-window_size, window_size+1)
    p_values = []
    for pos in positions:
        att_yes = df_filtered[(df_filtered['Position_relative_to_editing_site'] == pos) & (df_filtered['Edited'] == 'yes')]['Attention']
        att_no = df_filtered[(df_filtered['Position_relative_to_editing_site'] == pos) & (df_filtered['Edited'] == 'no')]['Attention']
        p_val = ttest_ind(att_yes, att_no, nan_policy='omit').pvalue if len(att_yes) > 1 and len(att_no) > 1 else np.nan
        p_values.append(p_val)
    plt.figure(figsize=(12, 7))
    sns.lineplot(x=positions, y=p_values, marker='o')
    plt.axhline(y=0.05, color='red', linestyle='--', label='Significance Threshold (0.05)')
    plt.title('Statistical Significance by Position (t-test)')
    plt.xlabel('Position Relative to Editing Site')
    plt.ylabel('p-value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "statistical_significance_by_position.png"))

    # Plot 8: Attention by Loop Size Grouped (Size = 1, 2, 3, >=4) and no N bases
    def loop_size_group(size):
        if size == 1:
            return '1'
        elif size == 2:
            return '2'
        elif size == 3:
            return '3'
        elif size >= 4 and size <= 6:
            return '4-6'
        else:
            return None

    loop_df = df_filtered[(df_filtered['Loop_Size'].notna()) & 
                          (df_filtered['Base'] != 'N') & 
                          (df_filtered['Loop_Partner_Base'] != 'N')].copy()
    loop_df['Loop_Size_Group'] = loop_df['Loop_Size'].apply(loop_size_group)

    plt.figure(figsize=(12, 7))
    sns.lineplot(data=loop_df, x='Position_relative_to_editing_site', y='Attention',
                 hue='Loop_Size_Group', marker='o', ci=None,
                 hue_order=['1', '2', '3', '4-6'])
    plt.axvline(x=0, color='red', linestyle='--', label='Editing Site')
    plt.title('Mean Attention by Loop Size Group')
    plt.xlabel('Position Relative to Editing Site')
    plt.ylabel('Mean Attention')
    plt.legend(title='Loop Size Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attention_by_loop_size_group.png"))

    print(f"All plots saved to {output_dir}")

# Function to identify loops in RNA structure
def identify_loops(sequence, structure):
    paired_bases = RNA.ptable(structure)
    loops = []
    i = 1
    loop_id = 0
    while i < len(sequence):
        if paired_bases[i] == 0:
            loop = []
            while i < len(sequence) and paired_bases[i] == 0:
                loop.append(i - 1)
                i += 1
            loops.append((loop_id, loop))
            loop_id += 1
        else:
            i += 1
    return loops

# Function to find loop pairings
def find_loop_pairing(sequence, structure):
    loops = identify_loops(sequence, structure)
    paired_bases = RNA.ptable(structure)
    loop_pairs = []
    base_to_loop_id = {}
    for loop_id, loop in loops:
        loop_info = {'loop_bases': [], 'loop_index': [], 'loop_pairs': [], 'pair_index': [], 'loop_id': loop_id}
        for i in loop:
            loop_info['loop_bases'].append(sequence[i])
            loop_info['loop_index'].append(i)
            base_to_loop_id[i] = loop_id
        min_number, max_number = min(loop), max(loop)
        min_number, max_number = min_number - 1, max_number + 1
        if min_number < 0 or max_number >= len(sequence):
            continue
        min_index, max_index = paired_bases[min_number+1]-1, paired_bases[max_number+1]-1
        if min_index < 0 or max_index < 0 or min_index == max_index:
            continue
        paired_seq_indices = list(range(min(min_index, max_index)+1, max(min_index, max_index)))
        loop_info['pair_index'] = paired_seq_indices
        loop_info['loop_pairs'] = [sequence[i] for i in paired_seq_indices]
        loop_pairs.append(loop_info)
    return loop_pairs, base_to_loop_id

def parse_csv_row(row):
    """
    Extracts the information from one CSV record.

    Expected columns in the CSV file:
        - structure   : Vienna secondary-structure string
        - L           : upstream sequence (left of the editing site)
        - R           : downstream sequence (right of the editing site)
        - yes_no      : 'yes' if the adenosine is edited, otherwise 'no'
    Returns:
        full_sequence      -> concatenated L + 'A' + R
        vienna_structure   -> value of 'structure'
        editing_site_idx   -> 0-based index of the adenosine within full_sequence
        edited_status      -> exactly 'yes' or 'no'
    """
    L_seq = row['L']
    if pd.isna(L_seq) or L_seq == "NA":
        L_seq = ""
    R_seq = row['R']
    if pd.isna(R_seq) or R_seq == "NA":
        R_seq = ""
    full_sequence = f"{L_seq}A{R_seq}"

    print(L_seq)

    editing_site_idx = len(L_seq)        # index of the edited adenosine
    vienna_structure = row['structure']

    # The column already contains 'yes' / 'no'; make sure the value is lowercase
    edited_status = str(row['yes_no']).strip().lower()   # -> 'yes' or 'no'

    return full_sequence, vienna_structure, editing_site_idx, edited_status

# Parse JSON line
def parse_json_line(json_line):
    data = json.loads(json_line)
    user_content = data['messages'][1]['content']
    L_seq = user_content.split('L:')[1].split(', A:')[0]
    A_seq = user_content.split('A:')[1].split(', R:')[0]
    R_seq = user_content.split('R:')[1].split(', Alu Vienna Structure:')[0]
    vienna = user_content.split('Alu Vienna Structure:')[1].strip()
    full_sequence = L_seq + A_seq + R_seq
    editing_site_idx = len(L_seq)
    edited_status = data['messages'][2]['content'].strip().lower()
    edited_status = 'yes' if 'yes' in edited_status else 'no'
    return full_sequence, vienna, editing_site_idx, edited_status

# Generate DataFrame with attention data
def map_bases_with_attention(sequence, vienna, editing_site_idx, attention_row):
    pairs = RNA.ptable(vienna)
    loops_info, base_to_loop_id = find_loop_pairing(sequence, vienna)
    loop_id_to_size = {loop['loop_id']: len(loop['loop_index']) for loop in loops_info}
    loop_partner_dict = {}
    for loop in loops_info:
        for idx, partner_idx in zip(loop['loop_index'], reversed(loop['pair_index'])):
            loop_partner_dict[idx] = partner_idx
    NNN_seq = 'N' * 10
    N_idx = sequence.find(NNN_seq)
    editing_site_side = 'left' if editing_site_idx < N_idx else 'right'
    opposite_editing_site_idx = pairs[editing_site_idx + 1] - 1
    if opposite_editing_site_idx is None or opposite_editing_site_idx < 0:
        opposite_editing_site_idx = loop_partner_dict.get(editing_site_idx, None)
    data = []
    for idx, base in enumerate(sequence):
        paired_idx = pairs[idx+1]-1 if pairs[idx+1] else None
        paired_base = sequence[paired_idx] if paired_idx is not None else None
        position_type = 'paired' if paired_idx is not None else 'loop/unpaired'
        loop_partner_idx = loop_partner_dict.get(idx, None)
        loop_partner_base = sequence[loop_partner_idx] if loop_partner_idx is not None else None
        pos_relative = idx - editing_site_idx #-1
        col_name = f'pos_{pos_relative}'
        attention_val = attention_row.get(col_name, None)
        if editing_site_side == 'left':
            same_side = idx <= N_idx
        else:
            same_side = idx >= N_idx + 10
        if same_side:
            position_side = 'same side'
            position_relative_to_editing = pos_relative #+ 1
        else:
            position_side = 'opposite side'
            position_relative_to_editing = idx - opposite_editing_site_idx if opposite_editing_site_idx is not None else None
        loop_id = base_to_loop_id.get(idx)
        loop_size = loop_id_to_size.get(loop_id) if loop_id is not None else None
        data.append({
            'Index': idx, 'Base': base, 'Paired_Index': paired_idx, 'Paired_Base': paired_base,
            'Loop_Partner_Index': loop_partner_idx, 'Loop_Partner_Base': loop_partner_base,
            'Position_Type': position_type, 'Attention': attention_val, 'Position_Side': position_side,
            'Position_relative_to_editing_site': position_relative_to_editing,
            'Loop_ID': loop_id, 'Loop_Size': loop_size
        })
    return pd.DataFrame(data)

# Main analysis function
def run_analysis(csv_file, attention_file, output_dir,window_size):
    attention_df = pd.read_csv(attention_file).drop(columns=['graph_index'], errors='ignore')
    site_df = pd.read_csv(csv_file)
    print(len(attention_df))
    print(len(site_df))
    all_sites_data = []
    for idx, row in site_df.iterrows():
        if idx >= len(attention_df):
            continue

        sequence, vienna_structure, editing_site_idx, edited_status = parse_csv_row(row)
        if len(sequence) != len(vienna_structure):
            print(f"[WARNING] site {idx+1}: sequence/structure length mismatch "
                f"({len(sequence)} vs {len(vienna_structure)}). Skipping.")
            continue
        attention_row = attention_df.iloc[idx].to_dict()
        result_df = map_bases_with_attention(sequence, vienna_structure, editing_site_idx, attention_row)
        result_df.insert(0, 'Edited', edited_status)
        result_df.insert(0, 'Site_Number', idx + 1)
        all_sites_data.append(result_df)
    final_df = pd.concat(all_sites_data, ignore_index=True)
    output_csv_path = os.path.join(output_dir, "structure_attentaion_info.csv")
    final_df.to_csv(output_csv_path, index=False)
    #generate_attention_plots(final_df, output_dir,window_size)
    create_composite_figure(output_csv_path, output_dir,window_size)
    print(f"Data analysis saved to {output_csv_path}")

# Command-line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RNA editing analysis script")
    parser.add_argument('--val_file', required= True, help="Input JSONL file")
    parser.add_argument('--attention', required= True, help="Attention CSV file")
    parser.add_argument('--window_size', type=int, default=50, help="Window size around the editing site (default: 50)")
    parser.add_argument('--output', required= True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    run_analysis(args.val_file, args.attention, args.output,args.window_size)

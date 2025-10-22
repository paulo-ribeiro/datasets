# -*- coding: utf-8 -*-
"""Discovery of Causal Model in Tumor Samples.ipynb

# **1) Input**

- The input for the method must be a MAF (Mutation Annotation Format) file, a tab-delimited file containing mutations found in tumor samples.

- The MAF columns utilized in this analysis are listed in the "maf_columns" object within the "Reading the MAF file" section.

<br>

### **Upload the input MAF file**

* Run the cell below and upload the MAF file using the button shown in the cell output.

* After uploading the input datasets, run the next cells.
"""

# D:\Projetos\Doutorado\Análises\2025.09 - Descoberta do modelo causal em amostras tumorais\Inputs\3 - MAFs com Colunas Filtradas
from google.colab import files
uploaded_maf_file = files.upload()
maf_file_name = list(uploaded_maf_file.keys())[0]

"""<br><br><br>
<br><br><br>

---

# **2) Initial configuration**

<br>

### **Install and import libraries**
"""

!pip install causal-learn

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
from causallearn.search.ConstraintBased.PC import pc
from scipy.stats import fisher_exact

import pydot
import re
from IPython.display import SVG
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.GeneralGraph import GeneralGraph

"""<br>

### **Initial configuration**
"""

def print_title(title): print("\n", title, "\n", "="*len(title), sep="")

# Output folders
dirs = {
    "tables": "Output/Tables",
    "graphs": "Output/Graphs",
    "graphs_all_genes": "Output/Causal graphs 1 - All genes",
    "graphs_known_drivers": "Output/Causal graphs 2 - Known drivers genes",
    "classified_edges": "Output/Causal graphs 3 - Classified edges"
}

# Create the output foldersfor path in dirs.values():
for path in dirs.values():
    os.makedirs(path, exist_ok=True)
    print(f"Directory '{path}' created or already exists.")

"""<br>

### **Load the known driver genes file**

* The file with known driver genes is automatically downloaded from the repository.
"""

#known_driver_file_name = "ncg_canonical_cancer_drivers.csv"
#!wget -O {known_driver_file_name} https://raw.githubusercontent.com/paulo-ribeiro/datasets/refs/heads/main/ncg_canonical_cancer_drivers.csv

known_driver_file_name = "Customized_gene_panel-Identification_somatic_variants_in_ctDNA.csv"
!wget -O {known_driver_file_name} https://raw.githubusercontent.com/paulo-ribeiro/datasets/refs/heads/main/Customized_gene_panel-Identification_somatic_variants_in_ctDNA.csv

"""<br>

### **Load the candidate cancer drivers file**

* The file with known driver genes is automatically downloaded from the repository.
"""

candidate_drivers_file_name = "ncg_candidate_drivers.csv"
!wget -O {candidate_drivers_file_name} https://raw.githubusercontent.com/paulo-ribeiro/datasets/refs/heads/main/ncg_candidate_drivers.csv

"""<br><br><br>
<br><br><br>

---

# **3) Functions**

### **Print information from the MAF data**

* Calculate and print information from the MAF data:

  - The number of unique tumor samples and mutations

  - The mean and median number of mutations per tumor sample
"""

def print_inf_maf_data(maf_df):
    # Obtaining all unique tumor samples from MAF
    num_unique_samples = maf_df["Tumor_Sample_Barcode"].unique()

    # Obtaining all unique mutations from MAF
    num_unique_mutations = maf_df["Hugo_Symbol"].unique()

    # Create the DataFrame with the number of mutations in each tumor sample
    number_mutations_df = maf_df.groupby("Tumor_Sample_Barcode").size().reset_index(name="Number of mutations")
    number_mutations_df = number_mutations_df.rename(columns={"Tumor_Sample_Barcode": "Sample"})

    # Calculate the mean and median of the number of mutations per sample
    mean_mutations = number_mutations_df["Number of mutations"].mean()
    median_mutations = number_mutations_df["Number of mutations"].median()

    # Print the results
    print_title("Information from the MAF data:")
    print(f"   Total mutations: {len(maf_df)}")
    print(f"   Number of unique tumor samples: {len(num_unique_samples)}")
    print(f"   Number of unique mutations: {len(num_unique_mutations)}")
    print(f"   Mean number of mutations per sample: {mean_mutations:.0f}")
    print(f"   Median number of mutations per sample: {median_mutations:.0f}")

"""<br>

### **Plot causal graph**
"""

def plot_causal_graph(cg, file_name="causal_graph", drivers=None,
                      candidate_drivers=None, legend=None, display_graph=True):

    if isinstance(cg, GeneralGraph):
        # Convert a graph object to a DOT object
        cg_dot = GraphUtils.to_pydot(cg)
        # Map with gene index and its name
        original_nodes = cg.get_nodes()
        node_map = {i: node.get_name() for i, node in enumerate(original_nodes)}
    elif isinstance(cg, pydot.Dot):
        cg_dot = copy.deepcopy(cg)
    else:
        raise ValueError("You need to provide a valid 'cg' object.")

    for node in cg_dot.get_nodes():
        # Setting the node font
        node.set_fontname("Helvetica")

        # Get the name of the gene
        node_name = node_map.get(int(node.get_name())) if isinstance(cg, GeneralGraph) else  node.get_attributes()['label'].strip('"')

        node.set_style("filled")
        node.set_color("transparent")

        # Coloring the nodes that are known/candidate driver genes
        if (drivers is not None) and (node_name in drivers):
            node.set_fillcolor("#9B59B6")
            node.set_fontcolor("white")
        elif (candidate_drivers is not None) and (node_name in candidate_drivers):
            node.set_fillcolor("#D9BDE1")
            node.set_fontcolor("#562E67")
        elif node_name.startswith("<<table"):
            node.set_fillcolor("white")
            node.set_fontcolor("black")
        else:
            node.set_fillcolor("#CCCCCC")
            node.set_fontcolor("#484848")

    # Adding the legend to the graph
    if (legend is not None):
        legend_node = create_legend(legend)
        cg_dot.add_node(legend_node)

    # Visualizing the graph
    if (display_graph):
        cg_dot.set_graph_defaults(dpi="60")
        img_svg = SVG(cg_dot.create_svg())
        print("\n\n")
        display(img_svg)

    # Saving the graph
    cg_dot.set_graph_defaults(dpi="200")
    cg_dot.write_png(file_name)
    print(f"\nCausal graph saved to {file_name}")

"""<br>

### **Print edges of causal graph**
"""

def print_edges(cg):
    print("\n")
    print_title("Edges of the causal graph")

    edges = cg.get_graph_edges()
    for edge in edges:
        print(f"   {edge}", "<<<", edge.__class__)

"""<br>

### **Print details of causal graph**
"""

def print_causal_graph_details(cg):
    print_title("Causal Graph Details")

    # Basic information
    print(f"   Number of nodes: {cg.G.get_num_nodes()}")
    print(f"   Total number of edges: {cg.G.get_num_edges()}\n")

    # Detailed edge counting
    edges = cg.G.get_graph_edges()
    directed_edges = 0
    undirected_edges = 0

    for edge in edges:
        if "->" in str(edge):
            directed_edges += 1
        elif "---" in str(edge):
            undirected_edges += 1

    print(f"   Directed edges: {directed_edges}")
    print(f"   Undirected edges: {undirected_edges}\n\n")

"""<br>

### **Add edges in graph**
"""

def add_edges_to_pydot(pydot_graph, edge_set, color="black", width=1.0, style="solid"):
    for edge_str in edge_set:
        nodes = re.split(r'\s*(-->|---|<->|o->|o-o)\s*', edge_str)

        if len(nodes) != 3: continue

        node1, arrowhead, node2 = nodes[0].strip(), nodes[1].strip(), nodes[2].strip()

        edge_attrs = {
            "color": color,
            "label": " " * 6,
            "penwidth": str(width),
            "style": style
        }

        if arrowhead == "-->":
            edge_attrs["dir"] = "forward"
        elif arrowhead == "<->":
            edge_attrs["dir"] = "both"
        elif arrowhead == "o->":
            edge_attrs["dir"] = "both"
            edge_attrs["arrowhead"] = "normal"
            edge_attrs["arrowtail"] = "odot"
        elif arrowhead == "o-o":
            edge_attrs["arrowhead"] = "odot"
            edge_attrs["arrowtail"] = "odot"
            edge_attrs["dir"] = "both"
        else:
            edge_attrs["dir"] = "none"

        pydot_graph.add_edge(pydot.Edge(node1, node2, **edge_attrs))

"""<br>

### **Convert edges of FCI graphs**
"""

def convert_edges_FCI(causal_graph_fci, title="", drivers=None, file_name="causal_graph_comparison.png"):
    node_names = causal_graph_fci.get_node_names()
    fci_edges_original = {str(edge) for edge in causal_graph_fci.get_graph_edges()}

    fci_edges_converted = set()
    for edge_str in fci_edges_original:
        converted_edge = edge_str
        if "o->" in converted_edge:
            converted_edge = converted_edge.replace("o->", "-->")
        if "o-o" in converted_edge:
            converted_edge = converted_edge.replace("o-o", "---")
        if "<->" in converted_edge:
            converted_edge = converted_edge.replace("<->", "---")
        fci_edges_converted.add(converted_edge)

    dot_graph = pydot.Dot(graph_type="graph", label=title, labelloc="t")

    for name in node_names:
        dot_graph.add_node(pydot.Node(name, label=name))

    # Adding edges to the graph
    add_edges_to_pydot(dot_graph, fci_edges_converted)

    return dot_graph, fci_edges_converted

"""<br>

### **Create legend for graph**
"""

def create_legend(items):
    rows = ""
    for color, desc in items:
        rows += f"<tr><td width='20' bgcolor='{color}'></td><td align='left'>{desc}</td></tr>"

    legend_html = f"<<table border='0' cellborder='0' cellspacing='0' cellpadding='3'>{rows}</table>>"

    legend_node = pydot.Node("legend", shape="plaintext", label=legend_html)
    legend_node.set_fontname("Helvetica")

    return legend_node

"""<br><br><br>
<br><br><br>

---

# **4) Read the input files**

### **MAF file**

- In this section, we read the MAF file and create a DataFrame containing only the columns used in this analysis.

- We also calculate the Variant Allele Frequency (VAF) of each mutation from the MAF and insert it into the DataFrame.

- Note: The VAF indicates the proportion of a variant allele in a gene in the cells of a sample.
"""

# MAF columns that will be in the dataframe
maf_columns = ["Hugo_Symbol", "Variant_Classification", "Variant_Type", "Tumor_Sample_Barcode",
               "t_ref_count", "t_alt_count", "n_ref_count", "n_alt_count"]

# Create the dataframe ignoring comment lines
maf_df = pd.read_csv(maf_file_name, sep="\t", usecols=maf_columns, comment="#")

# Calculate the depths
maf_df["n_depth"] = maf_df["n_ref_count"] + (maf_df["n_alt_count"])
maf_df["t_depth"] = maf_df["t_ref_count"] + (maf_df["t_alt_count"])

# Calculate the VAF
maf_df["VAF"] = maf_df["t_alt_count"] / (maf_df["t_ref_count"] + maf_df["t_alt_count"])

print_inf_maf_data(maf_df)
print("\n\n")
display(maf_df)

"""<br>

### **Known driver genes file**
"""

canonical_cancer_drivers = pd.read_csv(known_driver_file_name).iloc[:,0].to_list()
print(f"   Number of known driver genes: {len(canonical_cancer_drivers)}")

"""<br>

### **Candidate cancer drivers file**
"""

candidate_cancer_drivers = pd.read_csv(candidate_drivers_file_name).iloc[:,0].to_list()
print(f"   Number of candidate cancer drivers genes: {len(candidate_cancer_drivers)}")

"""<br><br><br>
<br><br><br>

---

# **5) Data preprocessing**

<br>

### **Filter mutations**

  - Filtering MAF mutations according to the following criteria:

    - Removal of mutations with missing values in critical columns.

    - Selection of only single nucleotide variants (SNVs).

    - Filtering based on tumor and normal read depth and allele counts.

    - Selection of only non-silent mutations with potential functional impact.
"""

#  Remove rows with missing values in critical columns
critical_columns = ["Hugo_Symbol", "t_ref_count", "t_alt_count"]
maf_df.dropna(subset=critical_columns, inplace=True)

# Filter by variant type (SNVs only)
maf_df = maf_df[maf_df["Variant_Type"] == "SNP"]

# Filter by number of reads and depth
# Tumor sample: ≥8 total reads, ≥3 alt reads OR VAF > 20%
# Normal sample: ≥6 total reads, ≤1 alt read OR VAF < 1%
maf_df = maf_df[
    ((maf_df["t_depth"] >= 8) &
     ((maf_df["t_alt_count"] >= 3) | (maf_df["VAF"] > 0.2))) &
    ((maf_df["n_depth"] >= 6) &
     ((maf_df["n_alt_count"] <= 1) | (maf_df["n_alt_count"]/(maf_df["n_ref_count"] + maf_df["n_alt_count"]) < 0.01)))
]

# Filter for non-silent mutations
non_silent_mutations = [
    "3'UTR", "5'UTR", "Frame_Shift_Del", "Frame_Shift_Ins",
    "In_Frame_Del", "In_Frame_Ins", "Missense_Mutation",
    "Nonsense_Mutation", "Nonstop_Mutation", "Splice_Site",
    "Translation_Start_Site"
]
maf_df = maf_df[maf_df["Variant_Classification"].isin(non_silent_mutations)]

"""<br>

### **Filter hypermutated samples**

  - Hypermutated samples should be excluded as they are often noisy outliers, which can distort analyses of clustering methods.
"""

# Count mutations per sample
mutations_per_sample = maf_df["Tumor_Sample_Barcode"].value_counts()

# Calculate Q1, Q3 and IQR
Q1 = mutations_per_sample.quantile(0.25)
Q3 = mutations_per_sample.quantile(0.75)
IQR = Q3 - Q1
threshold = min(Q3 + 4.5 * IQR, 600)

# Remove hypermutated samples from the dataframe
hypermutated_samples = mutations_per_sample[mutations_per_sample > threshold].index.tolist()
maf_df = maf_df[~maf_df["Tumor_Sample_Barcode"].isin(hypermutated_samples)]

"""<br>

### **Filter mutations in few samples**

  - Removing mutations present in an insufficient number of tumor samples.
"""

# Count samples per mutation
samples_per_mutation = maf_df["Hugo_Symbol"].value_counts()

# Set the minimum threshold of samples per mutation
min_samples = 10

# Filter mutations that have enough samples
valid_mutations = samples_per_mutation[samples_per_mutation >= min_samples].index.tolist()
maf_df = maf_df[maf_df["Hugo_Symbol"].isin(valid_mutations)]

"""<br>

### **Filter samples with few mutations**

  - Removing tumor samples with an insufficient number of mutations to ensure that clustering methods have enough data to identify meaningful patterns.
"""

# Count mutations per sample
mutations_per_sample = maf_df["Tumor_Sample_Barcode"].value_counts()

# Set the minimum threshold of mutations per sample
min_mutations = 10

# Filter samples that have enough mutations
valid_samples = mutations_per_sample[mutations_per_sample >= min_mutations].index.tolist()
maf_df = maf_df[maf_df["Tumor_Sample_Barcode"].isin(valid_samples)]

"""<br>

### **Data formatting - All genes**

* The dataset to be used by the causal discovery algorithms has the following structure:

   - **Columns**: most frequent genes in patients
   - **Rows**: patients
   - **Cells**: VAF of mutations
"""

mutations_counts = (
    maf_df.groupby("Hugo_Symbol")
          .size()
          .reset_index(name="count")
          .sort_values("count", ascending=False)
)

# Filter genes present in more samples
top_genes = mutations_counts.head(30)["Hugo_Symbol"].tolist()
top_genes_maf_df = maf_df[maf_df["Hugo_Symbol"].isin(top_genes)].copy().reset_index()

# Filter dataset columns for the causal discovery algorithm
vaf_mutations_df = top_genes_maf_df[["Hugo_Symbol", "Tumor_Sample_Barcode", "VAF"]]

# Create the dataset in the format required for the causal discovery algorithm
all_genes_data_df = vaf_mutations_df.pivot_table(index="Tumor_Sample_Barcode",
                                                 columns="Hugo_Symbol",
                                                 values="VAF",
                                                 aggfunc="mean")

# Convert dataset to binary: 1 if mutation exists (value is not NaN), 0 otherwise
all_genes_binary_data_df = all_genes_data_df.notna().astype(int)

# Create matrix with data for causal discovery algorithm
all_genes_data_matrix = all_genes_binary_data_df.to_numpy()

# List of gene names for the causal discovery algorithm
all_genes_labels = all_genes_binary_data_df.columns.tolist()

# Save dataframe to TSV file
all_genes_binary_data_df.to_csv(dirs["tables"] + "/all_genes_binary_data_df.tsv", sep="\t", index=False)

display(all_genes_binary_data_df)

"""<br>

### **Data formatting - Drivers genes**

* The dataset to be used by the causal discovery algorithms has the following structure:

   - **Columns**: most frequent genes in patients
   - **Rows**: patients
   - **Cells**: VAF of mutations
"""

# Filter the original dataset for the causal discovery algorithm
vaf_mutations_df = maf_df[["Hugo_Symbol", "Tumor_Sample_Barcode", "VAF"]]

# Filter genes present in more samples
top_genes = mutations_counts.head(100)["Hugo_Symbol"].tolist()
top_genes_maf_df = maf_df[maf_df["Hugo_Symbol"].isin(top_genes)].copy().reset_index()

# Filter dataset columns for the causal discovery algorithm
vaf_mutations_df = top_genes_maf_df[["Hugo_Symbol", "Tumor_Sample_Barcode", "VAF"]]

# Filter dataframe to keep only known driver genes
driver_genes_df = vaf_mutations_df[vaf_mutations_df['Hugo_Symbol'].isin(canonical_cancer_drivers)].copy()

print_title("Filtering by Known Driver Genes")
print(f"   Number of mutations before filtering: {len(vaf_mutations_df)}")
print(f"   Number of driver genes in the reference list: {len(canonical_cancer_drivers)}")
print(f"   Number of mutations after filtering: {len(driver_genes_df)}")
print(f"   Number of unique genes remaining: {driver_genes_df['Hugo_Symbol'].nunique()}\n\n")

# Create the dataset in the format required for the causal discovery algorithm
driver_genes_data_df = driver_genes_df.pivot_table(index="Tumor_Sample_Barcode",
                                                    columns="Hugo_Symbol",
                                                    values="VAF",
                                                    aggfunc="mean")

# Convert dataset to binary: 1 if mutation exists (value is not NaN), 0 otherwise
driver_genes_binary_data_df = driver_genes_data_df.notna().astype(int)

# Create matrix with data for causal discovery algorithm
driver_genes_data_matrix = driver_genes_binary_data_df.to_numpy()

# List of gene names for the causal discovery algorithm
driver_genes_labels = driver_genes_binary_data_df.columns.tolist()

# Save dataframe to TSV file
driver_genes_binary_data_df.to_csv(dirs["tables"] + "/driver_genes_binary_data_df.tsv", sep="\t", index=False)

display(driver_genes_binary_data_df)

"""<br><br><br>
<br><br><br>

---

# **6) Discovery of the causal model**

<br>

<br>

### **FCI - All genes**
"""

from causallearn.search.ConstraintBased.FCI import fci

# Running the Fast Causal Inference (FCI) algorithm for causal discovery
causal_graph_all_genes_fci, edges = fci(all_genes_data_matrix, node_names=all_genes_labels)
# causal_graph_all_genes_fci, edges = fci(all_genes_data_matrix, node_names=all_genes_labels, independence_test_method="kci")

# Plotting and printing the results
plot_causal_graph(causal_graph_all_genes_fci,
                  file_name=dirs["graphs_all_genes"] + "/1.causal_graph_all_genes_fci.png",
                  drivers=canonical_cancer_drivers, candidate_drivers=candidate_cancer_drivers,
                  display_graph=False)

# Convert graph edges
dot_graph_all_genes_convert_edges_fci, edges_converted_all_genes_fci = convert_edges_FCI(causal_graph_all_genes_fci, drivers=canonical_cancer_drivers)

# Graph legend items
legend_graph_all_genes = [("#9B59B6", "Known cancer drivers"), ("#D9BDE1", "Candidate cancer drivers")]

# Plotting the graph with the converted edges
plot_causal_graph(dot_graph_all_genes_convert_edges_fci,
                  file_name=dirs["graphs_all_genes"] + "/1.causal_graph_all_genes_fci(converted edges).png",
                  drivers=canonical_cancer_drivers, candidate_drivers=candidate_cancer_drivers,
                  legend=legend_graph_all_genes)

"""<br>

### **FCI - Driver genes**
"""

from causallearn.search.ConstraintBased.FCI import fci

# Running the Fast Causal Inference (FCI) algorithm for causal discovery
causal_graph_driver_genes_fci, edges = fci(driver_genes_data_matrix, node_names=driver_genes_labels)

# Plotting and printing the results
plot_causal_graph(causal_graph_driver_genes_fci,
                  file_name=dirs["graphs_known_drivers"] + "/1.causal_graph_driver_genes_fci.png",
                  display_graph=False)

# Convert graph edges
dot_graph_driver_genes_convert_edges_fci, edges_converted_driver_genes_fci = convert_edges_FCI(causal_graph_driver_genes_fci)

# Graph legend items
legend_graph_driver_genes = [("#9B59B6", "Known cancer drivers")]

# Plotting the graph with the converted edges
plot_causal_graph(dot_graph_driver_genes_convert_edges_fci,
                  file_name=dirs["graphs_known_drivers"] + "/1.causal_graph_driver_genes_fci(converted edges).png",
                  drivers=canonical_cancer_drivers, candidate_drivers=candidate_cancer_drivers,
                  legend=legend_graph_driver_genes)

"""<br>

### **GES - All genes**
"""

from causallearn.search.ScoreBased.GES import ges

# Running the Greedy Equivalence Search (GES) algorithm for causal discovery
causal_graph_all_genes_ges = ges(all_genes_data_matrix, node_names=all_genes_labels)

# Plotting and printing the results
plot_causal_graph(causal_graph_all_genes_ges['G'],
                  file_name=dirs["graphs_all_genes"] + "/2.causal_graph_all_genes_ges.png",
                  drivers=canonical_cancer_drivers, candidate_drivers=candidate_cancer_drivers,
                  legend=legend_graph_all_genes)

"""<br>

### **GES - Driver genes**
"""

from causallearn.search.ScoreBased.GES import ges

# Running the Greedy Equivalence Search (GES) algorithm for causal discovery
causal_graph_driver_genes_ges = ges(driver_genes_data_matrix, node_names=driver_genes_labels)

# Plotting and printing the results
plot_causal_graph(causal_graph_driver_genes_ges['G'],
                  file_name=dirs["graphs_known_drivers"] + "/2.causal_graph_driver_genes_ges.png",
                  drivers=canonical_cancer_drivers, candidate_drivers=candidate_cancer_drivers,
                  legend=legend_graph_driver_genes)

"""<br><br><br>
<br><br><br>

---

# **7) Consensus graphs**

<br>

### **Functions for creating consensus graph**
"""

# Normalize edges (ignore direction/type and order nodes alphabetically)
def normalize_edges(edges):
    normalized = set()
    for e in edges:
        parts = re.split(r'\s*(-->|<--|---|<->|o->|<-o|o-o)\s*', e)
        if len(parts) == 3:
            n1, _, n2 = parts
            ordered = sorted([n1.strip(), n2.strip()])
            normalized.add(f"{ordered[0]} --- {ordered[1]}")
    return normalized

"""<br>"""

# Identify similar edges that appear in both graphs (ignoring direction)
def get_norm_edges_in_both(edges_cg1, edges_cg2, edges_in_both):
    # Normalize edges (ignore direction/type and order nodes alphabetically)
    norm_to_edge_cg1 = normalize_edges(edges_cg1)
    norm_to_edge_cg2 = normalize_edges(edges_cg2)

    # Identify node pairs that appear in both graphs (ignoring direction)
    norm_edges_in_both = norm_to_edge_cg1.intersection(norm_to_edge_cg2)

    # Remove similar edges (orange) that already have identical edges (red)
    norm_edges_in_both_clean = set()
    for norm_edge in norm_edges_in_both:
        n1, n2 = [x.strip() for x in norm_edge.split('---')]
        # Check if this node pair already has an identical edge
        has_identical = any(
            (n1 in e and n2 in e) for e in edges_in_both
        )
        if not has_identical:
            norm_edges_in_both_clean.add(norm_edge)

    return norm_edges_in_both_clean

"""<br>"""

# Creates the consensus graph from two causal graphs
def create_consensus_graph(edges_cg1, edges_cg2, node_names, title=""):
    # Identify exactly identical edges (same direction and type)
    edges_in_both = edges_cg1.intersection(edges_cg2)

    # Identify similar edges that appear in both graphs (ignoring direction)
    norm_edges_in_both = get_norm_edges_in_both(edges_cg1, edges_cg2, edges_in_both)

    # Create the consensus graph
    consensus_pydot = pydot.Dot(graph_type="graph", label=title, labelloc='t')

    # Add nodes with label
    for name in node_names:
        consensus_pydot.add_node(pydot.Node(name, label=name))

    # Add identical edges (red)
    add_edges_to_pydot(consensus_pydot, edges_in_both, "#000000", 2.0)

    # Add edges with same nodes but different directions/types (orange)
    add_edges_to_pydot(consensus_pydot, norm_edges_in_both, "#666666", 2.0, style="dashed")

    print_title("Consensus Graph")
    print(f"   Edges in both graphs (identical): {len(edges_in_both)}")
    print(f"   Edges with same nodes but different types/directions: {len(norm_edges_in_both)}")

    return consensus_pydot

"""<br>

### **FCI and GES - All genes**
"""

fci_edges = edges_converted_all_genes_fci
ges_edges = {str(edge) for edge in causal_graph_all_genes_ges['G'].get_graph_edges()}

dot_consensus_graph_fci_converted_edges_ges = create_consensus_graph(fci_edges, ges_edges, all_genes_labels)

# Graph legend items
legend_consensus_graph = [("#000000", "Identical edges"), ("#666666", "Similar edges")]

plot_causal_graph(dot_consensus_graph_fci_converted_edges_ges,
                  file_name=dirs["graphs_all_genes"] + "/3.consensus_graph_fci(converted)_ges.png",
                  drivers=canonical_cancer_drivers, candidate_drivers=candidate_cancer_drivers,
                  legend=legend_consensus_graph)

"""<br><br><br>
<br><br><br>

---

# **8) Classifying edges**

- The edges of causal graphs are classified as "co-occurrence" and "mutual exclusivity"

<br>

### **Function to classify edges**

- The edges of causal graphs are classified as "co-occurrence" and "mutual exclusivity"
"""

import copy
from causallearn.graph.GeneralGraph import GeneralGraph

def classify_graph_edges(cg, binary_data_df, cooccurrence_threshold=2.0, exclusion_threshold=0.1):
    if isinstance(cg, GeneralGraph):
        # Convert a graph object to a DOT object
        cg_dot = GraphUtils.to_pydot(cg)

        # Map with gene index and its name
        original_nodes = cg.get_nodes()
        node_map = {i: node.get_name() for i, node in enumerate(original_nodes)}
    elif isinstance(cg, pydot.Dot):
        cg_dot = copy.deepcopy(cg)
    else:
        raise ValueError("You need to provide a valid 'cg' object.")

    # Get the edges of the causal graph
    causal_graph_edges = cg_dot.get_edges()

    # Number of edges that have been colored in the causal graph
    edges_modified = 0

    for edge in causal_graph_edges:
        # Get the names of the genes from the edge
        if isinstance(cg, GeneralGraph):
            source_name = node_map.get(edge.get_source())
            destination_name = node_map.get(edge.get_destination())
        else:
            source_name = edge.get_source()
            destination_name = edge.get_destination()

        # Create contingency table of edge genes
        contingency_table = pd.crosstab(binary_data_df[source_name], binary_data_df[destination_name])

        # Perform a Fisher exact test on the contingency table
        odds_ratio, p_value = fisher_exact(contingency_table)

        edge.set_color("gray40")
        edge.set_penwidth(1.5)

        # If the association is statistically significant
        if p_value < 0.05:
            if odds_ratio >= cooccurrence_threshold:
                # Strong co-occurrence
                edge.set_color("dodgerblue3")
                edge.set_penwidth(2.0)
                edges_modified += 1
            elif odds_ratio <= exclusion_threshold:
                # Strong mutual exclusivity
                edge.set_color("firebrick3")
                edge.set_penwidth(2.0)
                edges_modified += 1

    print(f"Total edges classified: {edges_modified}")

    return cg_dot

"""<br>

### **FCI - All Genes**
"""

# Classify the edges of the causal graph
dot_graph_all_genes_classified_edges_fci = classify_graph_edges(dot_graph_all_genes_convert_edges_fci, all_genes_binary_data_df)

# Graph legend items
legend_graph_classified_edges = [("#9B59B6", "Known cancer drivers"), ("#D9BDE1", "Candidate cancer drivers"),
     ("#FFFFFF", " "), ("dodgerblue3", "Co-occurrence"), ("firebrick3", "Mutual exclusivity")]

# Plotting the graph with the classified edges
plot_causal_graph(dot_graph_all_genes_classified_edges_fci,
                  file_name=dirs["classified_edges"] + "/1.causal_graph_all_genes_classified_converted_edges_fci.png",
                  drivers=canonical_cancer_drivers, candidate_drivers=candidate_cancer_drivers,
                  legend=legend_graph_classified_edges)

"""<br>

### **FCI - Drivers Genes**
"""

# Classify the edges of the causal graph
dot_graph_drivers_genes_classified_edges_fci = classify_graph_edges(dot_graph_driver_genes_convert_edges_fci, driver_genes_binary_data_df)

# Plotting the graph with the classified edges
plot_causal_graph(dot_graph_drivers_genes_classified_edges_fci,
                  file_name=dirs["classified_edges"] + "/2.causal_graph_driver_genes_classified_converted_edges_fci.png",
                  drivers=canonical_cancer_drivers, candidate_drivers=candidate_cancer_drivers,
                  legend=legend_graph_classified_edges)

"""<br>

### **GES - All Genes**
"""

# Classify the edges of the causal graph
dot_graph_all_genes_classified_edges_ges = classify_graph_edges(causal_graph_all_genes_ges["G"], all_genes_binary_data_df)

# Plotting the graph with the classified edges
plot_causal_graph(dot_graph_all_genes_classified_edges_ges,
                  file_name=dirs["classified_edges"] + "/3.causal_graph_all_genes_classified_edges_ges.png",
                  drivers=canonical_cancer_drivers, candidate_drivers=candidate_cancer_drivers,
                  legend=legend_graph_classified_edges)

"""<br>

### **GES - Drivers Genes**
"""

# Classify the edges of the causal graph
dot_graph_drivers_genes_classified_edges_ges = classify_graph_edges(causal_graph_driver_genes_ges["G"], driver_genes_binary_data_df)

# Plotting the graph with the classified edges
plot_causal_graph(dot_graph_drivers_genes_classified_edges_ges,
                  file_name=dirs["classified_edges"] + "/4.causal_graph_driver_genes_classified_edges_ges.png",
                  drivers=canonical_cancer_drivers, candidate_drivers=candidate_cancer_drivers,
                  legend=legend_graph_classified_edges)

"""<br>

### **FCI and GES - All genes**
"""

# Classify the edges of the causal graph
dot_consensus_graph_all_genes_classified_edges_fci_ges = classify_graph_edges(dot_consensus_graph_fci_converted_edges_ges, all_genes_binary_data_df)

# Plotting the graph with the classified edges
plot_causal_graph(dot_consensus_graph_all_genes_classified_edges_fci_ges,
                  file_name=dirs["classified_edges"] + "/5.consensus_graph_all_genes_classified_edges_fci_ges.png",
                  drivers=canonical_cancer_drivers, candidate_drivers=candidate_cancer_drivers,
                  legend=legend_graph_classified_edges)


Data Loading
============

Three types of benchmark datasets are provided:

- Residue prediction (AA): Datasets used to predict residue (amino acid) specific properties.
- Domain prediction (DOM): Dataset used to predict domain specific properties
- Sequence prediction (SEQ): Datasets used to predict sequence specific properties

The classification of each dataset is indicated as first part of their name followed by an abbreviation for the
specific dataset (e.g., 'AA_LDR', 'DOM_GSEC', 'SEQ_AMYLO'). For some datasets, an additional version of it is provided
for positive-unlabeled (PU) learning containing only positive (1) and unlabeled (2) data samples, as indicated by
*dataset_name*_PU (e.g., 'DOM_GSEC_PU').

See also

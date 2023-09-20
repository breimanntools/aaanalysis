Data Flow and Enry Points
=========================

The AAanalysis toolkit uses different DataFrames starting from DataFrames containing amino acid scales information
(df_scales, df_cat) or sequence information (df_seq), which can be modified to obtain specific sequence parts (df_parts).
Amino acid scales and sequence parts together with split settings are the input for the CPP algorithm, creating
various physicochemical features (df_feat) by comparing two sets of protein sequences.


import matplotlib.pyplot as plt
import aaanalysis as aa
sf = aa.SequenceFeature()
df_seq = aa.load_dataset(name='SEQ_DISULFIDE', min_len=100)
labels = list(df_seq["label"])
df_parts = sf.get_df_parts(df_seq=df_seq, jmd_n_len=10, jmd_c_len=10)

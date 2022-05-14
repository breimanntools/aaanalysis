"""
Script for Part objects used to retrieve sequence parts for given sequences.
"""
import aaanalysis.cpp._utils as ut


# I Helper Functions
# Checking functions
def check_input_part_creation(seq=None, tmd_start=None, tmd_stop=None):
    """Check if input for part creation is given"""
    if None in [seq, tmd_start, tmd_stop]:
        raise ValueError("'seq', 'tmd_start', 'tmd_stop' must be given (should not be None).")


def check_parts(tmd=None, jmd_n=None, jmd_c=None):
    """Check if parts are given"""
    if None in [tmd, jmd_n, jmd_c]:
        raise ValueError("'tmd', 'jmd_n', and 'jmd_c' must be given (should not be None)")


def check_ext_len(jmd_n=None, jmd_c=None, ext_len=None):
    """Check if lengths are matching"""
    if jmd_n is None or jmd_c is None:
        raise ValueError(f"'jmd_n' ({jmd_n}) and 'jmd_c' ({jmd_c}) should be given (not None)")
    if ext_len > len(jmd_n):
        raise ValueError(f"'ext_len' ({ext_len}) must be <= length of jmd_n ({len(jmd_n)})")
    if ext_len > len(jmd_c):
        raise ValueError(f"'ext_len' ({ext_len}) must be <= length of jmd_c ({len(jmd_c)})")


# Part helper functions
def _retrieve_string_starting_at_end(seq, start=None, end=None):
    """Reverse_string_start_end"""
    def reverse_string(s):
        return s[::-1]
    reversed_seq = reverse_string(seq)
    reversed_seq_part = reversed_seq[start:end]
    seq = reverse_string(reversed_seq_part)
    return seq


def _get_dict_part_seq_from_seq(tmd=None, jmd_n=None, jmd_c=None, ext_len=0):
    """Get dictionary for part to sequence

    Parameters
    ----------
    tmd: sequence of TMD
    jmd_n: sequence of JMD-N
    jmd_c: sequence of JMD-C
    ext_len: length of extending part (starting from C and N terminal part of TMD)

    Returns
    -------
    dict_part_seq: dictionary for each sequence part
    """
    tmd_n = tmd[0:round(len(tmd) / 2)]
    tmd_c = tmd[round(len(tmd) / 2):]
    ext_n = _retrieve_string_starting_at_end(jmd_n, start=0, end=ext_len)  # helix_stop motif for TMDs
    ext_c = jmd_c[0:ext_len]  # anchor for TMDs
    tmd_e = ext_n + tmd + ext_c
    part_seq_dict = {'tmd': tmd, 'tmd_e': tmd_e,
                     'tmd_n': tmd_n, 'tmd_c': tmd_c,
                     'jmd_n': jmd_n, 'jmd_c': jmd_c,
                     'ext_n': ext_n, 'ext_c': ext_c,
                     'tmd_jmd': jmd_n + tmd + jmd_c,
                     'jmd_n_tmd_n': jmd_n + tmd_n, 'tmd_c_jmd_c': tmd_c + jmd_c,
                     'ext_n_tmd_n': ext_n + tmd_n, 'tmd_c_ext_c': tmd_c + ext_c}
    return part_seq_dict


def _get_parts_from_df(df=None, entry=None):
    """Get features from df"""
    if not {"tmd", "jmd_n", "jmd_c", ut.COL_ENTRY}.issubset(set(df)):
        raise ValueError("'tmd', 'jmd_n', 'jmd_c' and '{}' must be given in df".format(ut.COL_ENTRY))
    if entry not in list(df[ut.COL_ENTRY]):
        raise ValueError("'{}' not in 'df'".format(ut.COL_ENTRY))
    tmd, jmd_n, jmd_c = df[df[ut.COL_ENTRY] == entry][["tmd", "jmd_n", "jmd_c"]].values[0]
    return tmd, jmd_n, jmd_c


class _PartsCreator:
    """Class for creating all sequence features necessary for CPP analysis:
        a) Target Middle Domain (TMD) or transmembrane domain for intramembrane proteases (IMP) substrates,
            whose length can vary between the protein sequences.
        b) Juxta Middle Domain (JMD) or juxtamembrane domain for IMP substrates,
            which flank the TMD N- and C-terminal with defined length.
    These sequence features can be derived from the total sequence, the TMD start and stop position,
        and the length of the JMD and the extended region (given for one side).

    Parameters
    ----------
    seq: sequence (e.g., amino acids sequence for protein)
    tmd_start: start position within the sequence of TMD
    tmd_stop: stop position within the sequence of the TMD
    jmd_n_len: length of JMD-N
    jmd_c_len: length of JMD-C
    """
    def __init__(self, seq=None, tmd_start=None, tmd_stop=None, jmd_n_len=10, jmd_c_len=10):
        check_input_part_creation(seq=seq, tmd_start=tmd_start, tmd_stop=tmd_stop)
        self._seq = seq
        self._tmd_start = tmd_start
        self._tmd_stop = tmd_stop
        self._jmd_n_len = jmd_n_len
        self._jmd_c_len = jmd_c_len
        self._n_terminus_len = self._tmd_start - 1
        self._c_terminus_len = len(self._seq) - self._tmd_stop
        self.tmd = self._tmd()
        self.jmd_n = self._jmd_n()
        self.jmd_c = self._jmd_c()

    def _tmd(self):
        """Get sequence of TMD"""
        start = self._tmd_start - 1
        stop = self._tmd_stop
        return self._seq[start:stop]

    def _jmd_n(self):
        """Get sequence of N-terminal JMD"""
        if self._n_terminus_len >= self._jmd_n_len:
            start = self._tmd_start - (self._jmd_n_len + 1)
            stop = self._tmd_start - 1
            jmd_n = self._seq[start:stop]
        else:
            start = 0
            stop = self._tmd_start - 1
            part = self._seq[start:stop]
            jmd_n = ut.STR_AA_GAP * (self._jmd_n_len - len(part)) + part  # Add "-" (gap) for missing AA in JMD
        return jmd_n

    def _jmd_c(self):
        """Get sequence of C-terminal JMD"""
        if self._c_terminus_len >= self._jmd_c_len:
            start = self._tmd_stop
            stop = self._tmd_stop + self._jmd_c_len
            jmd_c = self._seq[start:stop]
        else:
            start = self._tmd_stop
            stop = self._tmd_stop + self._c_terminus_len
            part = self._seq[start:stop]
            jmd_c = part + ut.STR_AA_GAP * (self._jmd_c_len - len(part))  # Add "-" (gap) for missing AA in JMD
        return jmd_c


# II Main Functions
class Parts:
    """Class for retrieving all sequence features necessary for CPP analysis:
        a) Target Middle Domain (TMD) or transmembrane domain for intramembrane proteases (IMP) substrates,
            whose length can vary between the protein sequences.
        b) Juxta Middle Domain (JMD) or juxtamembrane domain for IMP substrates,
            which flank the TMD N- and C-terminal with defined length.
        c) Extended TMD (TMD-E), which is an uncertainty region flanking the TMD (a shorter JMD version)
            or the TM-Helix anchoring region for IMP substrates with defined length.
    Notes
    -----
    TMD, JMD-N, JMD-C can be created and used to create a part sequence dictionary with further features
    """
    def __init__(self):
        pass

    @staticmethod
    def create_parts(seq=None, tmd_start=None, tmd_stop=None, jmd_n_len=10, jmd_c_len=10):
        """Create features object to retrieve TMD, JMD-N, JMD-C

        Parameters
        ----------
        seq: sequence (e.g., amino acids sequence for protein)
        tmd_start: start position within the sequence of TMD
        tmd_stop: stop position within the sequence of the TMD
        jmd_n_len: length of JMD-N
        jmd_c_len: length of JMD-C

        Returns
        -------
        parts: PartsCreator object to retrieve sequence for main parts (tmd, jmd_n, jmd_c)
        """
        parts = _PartsCreator(seq=seq, tmd_start=tmd_start, tmd_stop=tmd_stop,
                              jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len)
        return parts

    @staticmethod
    def get_dict_part_seq(df=None, entry=None, tmd=None, jmd_n=None, jmd_c=None, ext_len=0):
        """Get dictionary for part to sequence either (a) form df using entry or (b) from sequence.

        Parameters
        ----------
        df: df with sequence features
        entry: entry for which dict_part_seq should be created
        tmd: sequence of TMD
        jmd_n: sequence of JMD-N
        jmd_c: sequence of JMD-C
        ext_len: length of extending part (starting from C and N terminal part of TMD)

        Returns
        -------
        dict_part_seq: dictionary with parts to sequence of parts for given entry
        """
        if not (df is None or entry is None):
            tmd, jmd_n, jmd_c = _get_parts_from_df(df=df, entry=entry)
        check_parts(tmd=tmd, jmd_n=jmd_n, jmd_c=jmd_c)
        check_ext_len(jmd_n=jmd_n, jmd_c=jmd_c, ext_len=ext_len)
        dict_part_seq = _get_dict_part_seq_from_seq(tmd=tmd, jmd_n=jmd_n, jmd_c=jmd_c, ext_len=ext_len)
        return dict_part_seq

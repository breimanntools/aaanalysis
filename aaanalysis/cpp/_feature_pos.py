"""
This is a script for SequenceFeaturePositions object used to retrieve sequence positions
for given parts and features .
"""

from aaanalysis.cpp._part import Parts
from aaanalysis.cpp._split import Split
import aaanalysis.cpp._utils as ut


# I Helper Functions
def check_dict_part_pos(dict_part_pos=None):
    """Check if dict_part_pos is valid"""
    list_parts = list(dict_part_pos.keys())
    wrong_parts = [x for x in list_parts if x not in ut.LIST_ALL_PARTS]
    if len(wrong_parts) > 0:
        error = f"Following parts from 'dict_part_pos' are not valid: {wrong_parts}." \
                f"\n Parts should be as follows: {ut.LIST_ALL_PARTS}"
        raise ValueError(error)


def check_part_args_non_negative_int(tmd_len=20, jmd_n_len=10, jmd_c_len=10, ext_len=0, start=1):
    """Check if args non-negative integers"""
    ut.check_non_negative_number(name="start", val=start)
    args = zip(["tmd_len", "jmd_n_len", "jmd_c_len", "ext_len"],
               [tmd_len, jmd_n_len, jmd_c_len, ext_len])
    for name, val in args:
        ut.check_non_negative_number(name=name, val=val, min_val=0)


# II Main Functions
class SequenceFeaturePositions:
    """Class for getting sequence positions for features"""

    @staticmethod
    def get_dict_part_pos(tmd_len=20, jmd_n_len=10, jmd_c_len=10, ext_len=0, start=1):
        """Get dictionary for part to positions.

        Parameters
        ----------
        tmd_len: length of TMD
        jmd_n_len: length of JMD-N
        jmd_c_len: length of JMD-C
        ext_len: length of extending part (starting from C and N terminal part of TMD)
        start: position label of first position

        Returns
        -------
        dict_part_pos: dictionary with parts to positions of parts
        """
        check_part_args_non_negative_int(tmd_len=tmd_len, jmd_n_len=jmd_n_len, jmd_c_len=jmd_c_len,
                                         ext_len=ext_len, start=start)
        pa = Parts()
        jmd_n = list(range(0, jmd_n_len))
        tmd = list(range(jmd_n_len, tmd_len+jmd_n_len))
        jmd_c = list(range(jmd_n_len + tmd_len, jmd_n_len + tmd_len + jmd_c_len))
        # Change int to string and adjust length
        jmd_n = [i + start for i in jmd_n]
        tmd = [i + start for i in tmd]
        jmd_c = [i + start for i in jmd_c]
        dict_part_pos = pa.get_dict_part_seq(tmd=tmd, jmd_n=jmd_n, jmd_c=jmd_c, ext_len=ext_len)
        return dict_part_pos

    @staticmethod
    def get_positions(dict_part_pos=None, features=None, as_str=True):
        """Get list of positions for given feature names.

        Parameters
        ----------
        dict_part_pos: dictionary with parts to positions of parts
        features: list with feature ids
        as_str: bool whether to return positions as string or list

        Returns
        -------
        list_pos: list with positions for each feature in feat_names
        """
        check_dict_part_pos(dict_part_pos=dict_part_pos)
        features = ut.check_features(features=features, parts=list(dict_part_pos.keys()))
        sp = Split(type_str=False)
        list_pos = []
        for feat_id in features:
            part, split, scale = feat_id.split("-")
            split_type, split_kwargs = ut.check_split(split=split)
            f_split = getattr(sp, split_type.lower())
            pos = sorted(f_split(seq=dict_part_pos[part.lower()], **split_kwargs))
            if as_str:
                pos = str(pos).replace("[", "").replace("]", "").replace(" ", "")
            list_pos.append(pos)
        return list_pos


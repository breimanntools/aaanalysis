.. _usage_principles_aaontology:

AAontology: Classification of Amino Acid Scales
===============================================

Amino acid scales turn each residue into a number (hydrophobicity, charge, size, and so
on), but hundreds of scales exist and most of them measure overlapping properties.
AAontology is a two-level classification that organises these scales by *what they
actually capture*, introduced in [Breimann24b]_. It is what lets a CPP feature carry a
readable physicochemical meaning rather than an opaque scale identifier: every scale
belongs to a subcategory (for example *Polarity*) and a top-level category (for example
*ASA/Volume*), so a feature signature can be summarised by category and interpreted at a
glance.

.. admonition:: Provided by
   :class: note

   In AAanalysis the ontology ships with the data. :func:`~aaanalysis.load_scales`
   returns the scales (``df_scales``) together with their two-level classification
   (``df_cat``). Browse the full category and subcategory list in
   :ref:`Data Tables <tables>`, and see the :ref:`API reference <api>` for the loader
   options.

AAontology was created by automatic scale classification followed by manual refinement:

.. figure:: /_artwork/schemes/scheme_AAontology1.png

   Workflow of scale classification from [Breimann24b]_.

It comprises 586 amino acid scales, organised into 67 subcategories and 8 categories.
Grouping scales this way keeps the feature space interpretable: instead of ranking 586
raw scales, CPP can pick representatives per category (see :ref:`AAclust
<usage_principles_aaclust>`) and report a signature in terms of a handful of
physicochemical themes.

.. figure:: /_artwork/schemes/scheme_AAontology2.png

   AAontology two-level amino acid scale classification from [Breimann24b]_.

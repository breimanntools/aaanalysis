# AAanalysis module map (internal dataflow)

How the public subpackages **connect at runtime** — the canonical pipeline
`load → parts → CPP → model → explain → plot`. This is a **dataflow/composition**
map, **not an import graph**: the frontends are deliberately decoupled (only a
handful of cross-subpackage imports; everything shares the `aaanalysis.utils`
barrel and backend isolation is test-gated), so the connections below happen in
**user code passing DataFrames**, not in module imports.

Scope: this is the *internal* mental model. The *external* ecosystem (AAanalysis ↔
sklearn / SHAP / biopython / upstream descriptors) is a separate diagram
(README + Introduction, issue #210). A user-facing rendered version of this map is
owned by the docs-architecture epic **#106**; this file is the GitHub/agent source.

```mermaid
flowchart LR
  subgraph DH["data_handling"]
    LD["load_dataset"]
    LS["load_scales"]
    LF["load_features"]
    EP["EmbeddingPreprocessor"]
  end
  subgraph SA["seq_analysis"]
    AWS["AAWindowSampler"]
    AL["AAlogo"]
  end
  subgraph FE["feature_engineering"]
    AAC["AAclust"]
    SF["SequenceFeature / NumericalFeature"]
    CPP["CPP / CPPGrid"]
    CPPP["CPPPlot"]
  end
  subgraph PU["pu_learning"]
    DPU["dPULearn"]
  end
  subgraph XAI["explainable_ai (+_pro)"]
    TM["TreeModel"]
    SM["ShapModel (pro)"]
  end
  subgraph PD["protein_design"]
    AAM["AAMut / SeqMut"]
  end

  LD -->|df_seq| SF
  LD -->|df_seq| AWS
  AWS -->|windows / labels| SF
  LS -->|df_scales| AAC
  AAC -->|reduced scale set| CPP
  LS -->|df_scales| CPP
  SF -->|df_parts| CPP
  CPP -->|df_feat| TM
  CPP -->|df_feat| CPPP
  LF -->|df_feat ref| CPPP
  DPU -->|labels| TM
  EP -->|embeddings X| TM
  TM -->|fitted model| SM
  SM -->|SHAP values| CPPP
  CPP -->|feature impact| AAM
  AL -.->|sequence logos| CPPP
```

**Cross-cutting (used by many, not a pipeline stage):** `plotting`
(`plot_settings`/colors/`plot_legend`/`plot_rank`) styles every `*Plot`; `metrics`
(`comp_*`) scores model/clustering outputs.

**Pro / utility subpackages (off the core flow):** `data_handling_pro`
(StructurePreprocessor, AnnotationPreprocessor), `seq_analysis_pro`
(comp_seq_sim, filter_seq, scan_motif), `explainable_ai_pro` (ShapModel),
`show_html` (display_df, dev).

<!-- MAP-SUBPKGS:START — roster checked by check_module_map.py; regenerate with --write-roster -->
- data_handling
- data_handling_pro
- explainable_ai
- explainable_ai_pro
- feature_engineering
- metrics
- plotting
- protein_design
- pu_learning
- seq_analysis
- seq_analysis_pro
- show_html
<!-- MAP-SUBPKGS:END -->

_Validated by `agent-readiness-audit` (`scripts/check_module_map.py`): every public
subpackage must appear above; the diagram/prose is curated by hand (semantic
dataflow can't be auto-derived)._

# MEG Coreference Experiment

All code related to an experiment using magnetoencephalography (MEG) to investigate human processing of coreference.

## Stimulus Generation

- Source data for generating stimuli are provided in the `stim_src` directory.
Outputs from stimulus generation scripts are saved in the `stim_gen` directory.

- Select names from Newman et al. (2018) norms:

        python -m meg_coref.select_names

- Collect trigram continuations for each predicate type (is a, owns a, likes to) from a KenLM model.
Requires a KenLM trigram model trained on Gigaword 3, which is too large to distribute with this repository.
The model can be shared upon request and must be placed at the path `stim_src/gigaword.3.kenlm`:

        python -m meg_coref.get_pred_trigrams
    
- Select predicate candidates based on study frequency and part-of-speech criteria:
    
        python -m meg_coref.extract_meg_predicates

- Filter out predicates that are deemed subjectively unnatural in the stimulus frame.
The filter must be constructed by hand.
Our filter is provided at `stim_src/filter.txt`:
    
        python -m meg_coref.filter_preds 

- Optimize selection of predicates to maximize cosine distance:

        python -m meg_coref.optimize_predicates
        
- Generate item stubs from selected names, predicates, and implicit causality verbs.
Our particular selections of predicates are hard-coded and will need to be overwritten if different predicates are selected.

        python -m meg_coref.generate_sentence_stubs
        
    The remaining elements of the experimental items (e.g. adverbial phrases, final sentence continuations, etc.) must be written by hand.
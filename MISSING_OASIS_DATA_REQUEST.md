# Missing OASIS-3 Data Request — NeuroFEDX

The current authorized OASIS-3 export used for this reproduction does not
contain five paper variables:

- TRAILARR — Trail Making A number of errors
- TRAILALI — Trail Making A number of correct lines
- TRAILBRR — Trail Making B number of errors
- TRAILBLI — Trail Making B number of correct lines
- MEMTIME — time elapsed between LOGIMEM and MEMUNITS administration

These are real UDS variables, not invented names. NACC documentation identifies
TRAILARR/TRAILALI/TRAILBRR/TRAILBLI as Trail Making sub-items and MEMTIME as
the elapsed time for the Logical Memory test.

## Exact request

Please provide an authorized OASIS-3/UDS export (or the source data package
used to create the current OASIS-3 clinical/psychometric CSV) that contains
the raw UDS C1 cognitive-assessment variables:

TRAILARR, TRAILALI, TRAILBRR, TRAILBLI, MEMTIME

If the export uses a different naming/version convention, provide the
corresponding data dictionary/codebook and the raw columns for the same
constructs. Do not substitute `lmdelay` for MEMTIME.

## Reproduction rule

Until those variables are supplied, the five features remain `UNAVAILABLE`.
The model must not silently impute or rename another variable into these
slots and call the paper's 34-feature set reproduced.

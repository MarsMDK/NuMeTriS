![PyPI](https://img.shields.io/badge/pypi-v2.1.1-blue)  [![License:GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10-blue) ![arXiv]([https://img.shields.io/badge/pypi-v2.1.1-blue](https://img.shields.io/badge/arXiv-2305.12179-orange))

# NuMeTriS: NUll ModEls for TRIadic Structures

NuMeTriS is a package developed on python3 for Pattern Detection of the 13 triadic connected sub-graphs using maximum-entropy models using direected and reciprocated constraints on network data.

NuMeTriS provides solvers for the binary models DBCM and RBCM, and the conditional weighted models CReMa and CRWCM.
All of these models are explained in [1](forthcoming on ArXiv).
The use of DBCM and CReMa enable the user to explicitly constrain network properties based on direction, such as out-degree, in-degree (binary) and out-strength and in-strength (weighted).
In contrast, the use of RBCM and CRWCM enable the user to constrain network properties based on both direction and reciprocity, such as the reciprocated and non-reciprocated degrees and the reciprocated and non-reciprocated strengths.

Moreover, after solving the models it is possible to generate the related ensembles and compute triadic occurrences, the arithmetic mean of the weights on triadic structures (average fluxes) and their geometric mean (intensities). While triadic occurrences and fluxes are explained in [1](forthcoming on ArXiv), the triadic intensity is the geometric mean of the weights in triadic structures, as explained in [2](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.71.065103) but here defined for all 13 sub-graphs.

To explore Maximum-Entropy modeling on networks, checkout [Maximum Entropy Hub](https://meh.imtlucca.it/).

When using the module for your scientific research please consider citing:


```
    @misc{divece2023commodityspecific,
      title={Commodity-specific triads in the Dutch inter-industry production network}, 
      author={Marzio Di Vece and Frank P. Pijpers and Diego Garlaschelli},
      year={2023},
      eprint={2305.12179},
      archivePrefix={arXiv},
      primaryClass={physics.soc-ph}
}

```
#### Contents
- [NuMeTriS: NUll ModEls for TRIadic Structures](#dygys-dyadic-gravity-regression-models-with-soft-constraints)
      - [Contents](#contents)
  - [Currently Available Models](#currently-available-models)
  - [Installation](#installation)
  - [Dependencies](#dependencies)
  - [How-to Guidelines](#how-to-guidelines)
    - [Class Instance and Empirical Network Statistics](#class-instance-and-empirical-network-statistics)
    - [Solving the models](#solving-the-models)
    - [Numerically Computing z-scores and significance profiles](#numerically-computing-z-scores-and-significance-profiles)
  - [Documentation](#documentation)
  - [Credits](#credits)

##  Currently Available Models
NuMeTriS contains models for triadic motif detection on network data for continuous-valued semi-definite positive weights.
The available models are:
* **DBCM** *Directed Binary Configuration Model* 
* **RBCM** *Reciprocated Binary Configuration Model* 
* **DBCM+CReMa** *Mixture model of Directed Binary Configuration Model and Conditional Reconstruction Method A* 
* **RBCM+CRWCM** *Mixture model of Reciprocated Binary Configuration Model and Conditionally Reciprocal Configuration Model* 

Please refer to the paper for further details.

## Installation
DyGyS can be installed via [pip](https://pypi.org/project/NuMeTriS/). You can do it from your terminal
```
    $ pip install NuMeTriS
```
If you already installed the package and want to  upgrade it,
you can type from your terminal:

```
    $ pip install NuMeTriS --upgrade
```

## Dependencies
NuMeTris uses the following dependencies:
* **scipy** for optimization and root solving of some of the models;
* **numba** for fast computation of network statistics and criterion functions.
* **matplotlib** in order to plot z-score profiles and significance profiles.

They can be easily installed via pip typing

    $ pip install scipy
    $ pip install numba
    $ pip install matplotlib


## How-to Guidelines
The module containes the class Graph, initiated with a 2D weighted adjacency matrix, explaining the interactions present in the network data.

### Class Instance and Empirical Network Statistics
To inizialize a Graph instance you can type:

    G = Graph(adjacency=Wij)

where Wij is the weighted adjacency matrix in 2-D numpy array format.

After initializing you can already explore core network statistics such as (out-)degree, in-degree, reciprocated degrees, non-reciprocated out-degrees and non-reciprocated in-degrees, available using the respective codewords:

    G.dseq_out, G.dseq_in, G.dseq_right, G.dseq_left, G.dseq_rec

or weighted core network statistics such as (out-)strength, in-strength, reciprocated strengths, non-reciprocated strengths, available using the respective codewords:

    G.stseq_out, G.stseq_in, G.stseq_right, G.stseq_left, G.stseq_rec_out, G.stseq_rec_in

Also triadic network statistics are computed, namely Occurrences, Intensities and Fluxes, available using the respective codewords:

    G.Nm_emp, G.Im_emp, G.Fm_emp


### Solving the models
You can explore the currently available models using
    
    G.implemented_models
use their names as described in this list not to incur in error messages.

When ready you can choose one of the aforementioned models and solve for their parameters using
    
    G.solver(model= <chosen model>)

Once you solved the model various other attributes become visible and measures dependent solely on criterion functions are computed. These include Loglikelihood, Jacobian, Infinite Jacobian Norm, relative error and relative infinite norm, available using the codewords:

    G.ll, G.jacobian, G.norm, G.relative_error, G.norm_rel

For further details on the .solve functions please see the documentation.



### Numerically Computing z-scores and significance profiles

Computing z-scores and significance profiles is very easy. 
Generating the network ensemble is very easy. It's enough to type:
    
    G.numerical_triadic_zscores(n_ensemble=<wanted number of graphs>,percentiles=<wanted percentiles for z-score CIs.>)

This routine generate "n_ensemble" graphs and computes z-scores for triadic occurrences (if a binary model is chosen) or triadic occurrences, fluxes and intensities (if a mixture model is chosen). Moreover it estimates a confidence interval for the z-score to understand the range of the values assumed in the ensemble. Percentiles are defined as tuple types and a 95% confidence interval can be estimated by the tuple (z_{2.5},z_{97.5}), for percentiles = (2.5,97.5).


This method returns expected triadic occurrences, triadic fluxes and triadic intensities 

    G.avg_Nm, G.avg_Fm, G.avg_Im,

corresponding z-scores

    G.zscores_Nm, G.zscores_Fm, G.zscores_Im,

inferior and superior percentiles

    G.zscores_down_Nm, G.zscores_down_Fm, G.zscores_down_Im, G.zscores_up_Nm, G.zscores_up_Fm, G.zscores_up_Im

and significance scores

    G.normalized_z_Nm, G.normalized_z_Fm, G.normalized_z_Im.

### Plot z-scores and significance profiles

Routine to plot z-scores and significance profiles. It's enough to type:

    G.plot_zscores(type='z-scores')
    G.plot_zscores(type='significance')
    G.plot_zscores(type='both')

It plots triadic occurrences (if a binary model is used) or occurrences, fluxes and intensities (if a mixture model is used).
It is possible to plot z-scores using the argument type='z-scores', significance profiles using argument type='significance' or both, using type='both'.



## Documentation
You can find the complete documentation of the DyGyS library in [documentation](https://numetris.readthedocs.io/en/latest/index.html)

## Credits

*Author*:

[Marzio Di Vece](https://www.imtlucca.it/it/marzio.divece) (a.k.a. [MarsMDK](https://github.com/MarsMDK))

*Acknowledgments*:
The module was developed under the supervision of [Diego Garlaschelli](https://www.imtlucca.it/en/diego.garlaschelli) and [Frank P. Pijpers](https://www.uva.nl/profiel/p/i/f.p.pijpers/f.p.pijpers.html).
It was developed at [IMT School for Advanced Studies Lucca](https://www.imtlucca.it/en) and [Statistics Netherlands](https://www.cbs.nl/en-gb) and
supported by the Italian ‘Programma di Attività Integrata’ (PAI) project ‘Prosociality, Cognition and Peer Effects’ (Pro.Co.P.E.), funded by IMT School for Advanced Studies.

# Accelerated Spatial Meta-Kriging (ASMK)

This package provides the principal functions to perform accelerated meta-kriging, for both univariate and multivariate spatial regression. In order to guarantee the reproducibility of scientific results, in this repository are also available all the scripts of code used for simulations and data analysis of **Luca Presicce** and Sudipto Banerjee (2024+) *"Accelerated Meta-Kriging for massive Spatial dataset"*.


--------------------------------------------------------------------------------
## Roadmap

| Folder | Description |
| :--- | :---: |
| `reproducibility-code` | contains simulation code & data analyses |
| `R` | contains funtions in R |
| `src` | contains function in Rcpp/C++ |

--------------------------------------------------------------------------------
## Guided installation
Since the package is not already available on CRAN, to install we use the `devtools` R package. Then, check for its presence on your device, otherwise install it:
```{r, echo = F, eval = F, collapse = TRUE}
if (!require(devtools)) {
  install.packages("devtools", dependencies = TRUE)
}
```
Once you have installed *devtools*, we can proceed. Let's install the `ASMK` package!
```{r}
devtools::install_github("lucapresicce/ASMK")
```
Cool! You are ready to start, now you too could perform **_fast & feasible_** Bayesian geostatistical modeling!

<!--
## Tutorial for usage
-->

--------------------------------------------------------------------------------
## Contacts

| | |
| :--- | :---: |
| Author | Luca Presicce (l.presicce@campus.unimib.it) |
| Maintainer | Luca Presicce (l.presicce@campus.unimib.it) |
| Reference | **Luca Presicce** and Sudipto Banerjee (2024+) *"Accelerated Meta-Kriging for massive Spatial dataset"*  |

<!--
Maintainer: l.presicce@campus.unimib.it
Reference: **Luca Presicce** and Sudipto Banerjee (2024+) *"Accelerated Meta-Kriging for massive Spatial dataset"* 
-->


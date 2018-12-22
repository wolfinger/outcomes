# outcomes

## about

the outcomes project's goal is to enable better data anlaysis for finance. it aims to:

1) minimize 'setup' redundency when starting a new research/analysis project
2) enable manipulation of data more easily and faster than in Excel
3) improve the robustsness of an analysis (e.g., analyzing distributions of rolling asset returns)

two primary uses for this python package:

1) analyzing returns data (calculating % change time series values)

    the following nomenclature is used for returns data structure:

        series types:
            topic: source data series (e.g., an asset's price level,
                an index's total return level, a trading strategy, etc.)
            rf: the risk-free asset
            bmk: the benchmark
            excess: the topic series over/under the risk-free series
            active: the topic series over/under the benchmark series

        calculated value types:
            ln: log (base e) values
            cmp: compound values

2) analyzing market data (using abosulte value data; e.g., bond yields)

    this analysis leverages the same data structure as the returns data;
    however, fields where a market data analysis is nonsensical
    are empty (e.g., Sharpe ratio)


## installation

[TODO]


## example usage

[TODO]


## release history

* unreleased
    * logically structure everything; kinda a hack job right now
    * documentation (ongoing effort to make this useful to devs)
    * front end (eventual effort to make this useful to normals)
    * turn time periods and data structure into classes
    * allow for passing in compound or log returns
    * build the drawdown algo using my brain instead of brute force
    * annualize and de-annualize functions

* 0.0.4 - 2018-12-22
    * NOTES
        * Plan (hopefully) for next version to be materially re-designed. I've hacked together this from my own internal use and to make it useful for others requires logical restructuring of things.
        * If you're a dev, this implies things will break next release...
    * ADD
        * Test script (although doesn't use unittest or similar framework)
    * CHANGE
        * change_analysis function can now take dynamic time period and measures
          lists
        * Misc clean-up of documentation
        * Corrected README v0.0.3 release date
        * Some lines are longer than 80 chars...style police shoot me

* 0.0.3 - 2018-12-13
    * ADD
        * Sphinx documentation (in 'docs' folder)
    * CHANGE
        * Function variable names to be more consistent / descriptive

* 0.0.2 - 2018-12-12
    * ADD
        * documentation (docstrings, better README, etc.)
    * CHANGE
        * file structure to better follow python norms

* 0.0.1 - 2018-04-04
    * work in progress


## meta

greg wolfinger - [@direwolfinger](https://twitter.com/direwolfinger) - greg@wolfinger.io

distributed under the MIT license. see ''LICENSE'' for more information.

[https://github.com/wolfinger/outcomes](https://github.com/wolfinger/outcomes)


## contributing

[TODO]
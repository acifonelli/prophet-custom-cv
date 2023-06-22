# Custom Cross-validation function for `Prophet: Automatic Forecasting Procedure`
## Original repository
    
The original code, together with instructions to install the package, is available at [Prophet: Automatic Forecasting Procedure](https://github.com/facebook/prophet).    
    
Original paper: [Sean J. Taylor, Benjamin Letham (2018) Forecasting at scale. The American Statistician 72(1):37-45](https://peerj.com/preprints/3190.pdf).    

## Custom cross-validation function

The default cross_validation function available in `Prophet` produces daily date range by default. Internally `Timedelta`s are used to generate the cutoffs instead of the daterange employed in the `make_future_dataframe()` function. `Timedelta` indeed expects a different set of frequencies, raging from weeks to nanoseconds.    
This is not suitable for all the dataset we can work with. In the `Airpassengers` dataset, for example, the number of observed passengers are aggregated on the first day of each month and we would like to forecast the number of passengers for the next `horizon` months - with `period` - and keep the first day of the month as a reference for the time index.    
A combination of horizon and period expressed in weeks or days will generate a misalignment.    
    
We introduce a custom cross-validation function - mirroring the original one - which preserves the `model.history` frequency while generating the cross-validation `Dataframe`.

## Examples

In the `notebooks` folder you can find an example on the `Airpassengers` dataset forecasting directly over a monthly horizon instead of daily.
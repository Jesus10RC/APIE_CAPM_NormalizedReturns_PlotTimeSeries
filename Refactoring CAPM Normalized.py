"""
Refactoring CAPM

"""
import importlib


import stream_functions
importlib.reload(stream_functions)
import stream_classes
importlib.reload(stream_classes)

#Input Parameters 
ric = 'SAN.MC' # SAN.MC  BBVA.MC  VWS.CO  MXN=X  AMZN  ^VIX  #y
benchmark = '^STOXX50E'  #^STOXX50E  ^STOXX  ^S&P  ^NASD  ^IPC  ^CAC40   #x

##Optinal - view h&tsp in f&c
x, x_str, t = stream_functions.load_time_series(ric)
jb = stream_classes.jarque_bera_test(x, x_str)
jb.compute()
# print(jb)
stream_functions.plot_timeseries_price(t, ric)
stream_functions.plot_histogram(x, x_str, jb.plot_str())

#In class CAPM -> x, y, t = stream_functions.synchronize_timeseries(ric, benchmark)
capm = stream_classes.capm_manager(ric, benchmark)
capm.load_timeseries()
capm.compute()
capm.scatterplot()
print(capm)

#Plot timeseries of ric and benchmark with different vertical axes
capm.plot_normalized()

#Plot with 2 vertical axes
capm.plot_dual_axes()

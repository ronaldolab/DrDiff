import numpy as np

from bokeh.plotting     import figure, output_file, show
from bokeh.embed        import components
from bokeh.layouts      import gridplot
from bokeh.models       import ColumnDataSource, Whisker
#from bokeh.resources    import CDN
#from bokeh.io           import export_svgs # needs npm install -g phantomjs-prebuilt; pip install selenium phantomjs pillow


# Plot the results to be embeded in the software html page
#def plot_results(rID, filename):
def plot_results(rID, path, Q_zero, Q_one):

    TOOLS = "save,pan,wheel_zoom,box_zoom,reset,box_select,lasso_select,crosshair"

    p_width=450
    p_height=350

    runId = rID # for testing
    #traj_folder = "/trajectories/" + str(runId) + "_"
    #out_folder  = "/outputs/" + str(runId) + "_"
    out_folder  = path + str(runId) + "_"

    # Plot F(Q) from -lnP
    file_FQ = out_folder + "Free_energy.dat"
    fFQ = open(file_FQ, 'r')
    F = np.loadtxt(fFQ.name, usecols=(0, 1), skiprows=0)
    x0 = F[:,0]
    y0 = F[:,1]
    fFQ.close()

    # Plot F(Q) from F_stochastic
    file_Fs =  out_folder + "F_stoch_Q.dat"
    fFs = open(file_Fs, 'r')
    F = np.loadtxt(fFs.name, usecols=(0, 1, 2), skiprows=0)
    x1    = F[:,0]
    y1    = F[:,1]
    y1err = F[:,2]
    fFs.close()

    # Transition State rectangle
    top     = np.max(y1)
    bottom  = np.min(y1)

    pFQ = figure(tools=TOOLS, plot_width=p_width, plot_height=p_height, title="Free energy from -log(histogram[Q]) and by using D(Q) and v(Q)", x_axis_label='reaction coordinate, Q', y_axis_label='Free energy, F(Q)')


    # add renderers
    # TS
    pFQ.quad(top=top, bottom=bottom, left=Q_zero,
      right=Q_one, color=(0,176,246,0.2))
    # F
    pFQ.circle(x0, y0, color="red", fill_color="white", size=8, alpha=0.2, legend="F_equilibrium")
    pFQ.line(x0, y0, color="red")
    # F_stoch
    pFQ.circle(x1, y1, color="green", fill_color="white", size=8, alpha=0.2, legend="F_stochastic")
    pFQ.line(x1, y1, color="green",)

    # add renderers
    # D and v
    # Plot F(Q) from -lnP
    file_D = out_folder + "DQ.dat"
    fD = open(file_D, 'r')
    F = np.loadtxt(fD.name, usecols=(0, 1), skiprows=0)
    x2 = F[:,0]
    y2 = F[:,1]
    #y2err = F[:,2]
    fD.close()

    # Plot F(Q) from F_stochastic
    file_v = out_folder + "VQ.dat"
    fv = open(file_v, 'r')
    F = np.loadtxt(fv.name, usecols=(0, 1, 2), skiprows=0)
    x3    = F[:,0]
    y3    = F[:,1]
    #y3err = F[:,2]
    fv.close()

    # Transition State rectangle
    top      = np.max(y2)
    bottom   = np.min(y2)

    pD = figure(tools=TOOLS, plot_width=p_width, plot_height=p_height, title="Diffusion coefficient, D(Q)", x_axis_label='reaction coordinate, Q', y_axis_label='D(Q)')

    # TS
    pD.quad(top=top, bottom=bottom, left=Q_zero,
      right=Q_one, color=(0,176,246,0.2))

    # D
    pD.circle(x2, y2, color="green", fill_color="white", size=8, alpha=0.2, legend="D")
    pD.line(x2, y2, color="green")

    # Transition State rectangle
    top      = np.max(y3)
    bottom   = np.min(y3)

    pV = figure(tools=TOOLS, plot_width=p_width, plot_height=p_height, title="Drift-velocity coefficient, v(Q)", x_axis_label='reaction coordinate, Q', y_axis_label='v(Q)')

    # TS
    pV.quad(top=top, bottom=bottom, left=Q_zero,
      right=Q_one, color=(0,176,246,0.2))

    # V
    pV.circle(x3, y3, color="green", fill_color="white", size=8, alpha=0.2, legend="v")
    pV.line(x3, y3, color="green")


    # Dump the subplots in a gridplot
    p = gridplot([[pFQ],[pD],[pV]])


    # It will return the tuple script, div
    return (components(p))

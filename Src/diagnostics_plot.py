
def diagnostic_plot(model_fit, df_name, response_variable):
    import numpy as np
    import pandas as pd

    import seaborn as sns
    import matplotlib.pyplot as plt

    import statsmodels.formula.api as smf
    from statsmodels.graphics.gofplots import ProbPlot

    plt.style.use('seaborn') # pretty matplotlib plots
    plt.rc('font', size=14)
    plt.rc('figure', titlesize=18)
    plt.rc('axes', labelsize=15)
    plt.rc('axes', titlesize=18)


    model_fitted_y = model_fit.fittedvalues

    model_residuals = model_fit.resid

    model_norm_residuals = model_fit.get_influence().resid_studentized_internal

    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))

    model_abs_resid = np.abs(model_residuals)

    model_leverage = model_fit.get_influence().hat_matrix_diag

    model_cooks = model_fit.get_influence().cooks_distance[0]


    plot_lm_1 = plt.figure(1)
    plot_lm_1.set_figheight(4)
    plot_lm_1.set_figwidth(6)

    plot_lm_1.axes[0] = sns.residplot(model_fitted_y, response_variable, data=df_name,
                                      lowess=True,
                                      scatter_kws={'alpha': 0.5},
                                      line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    plot_lm_1.axes[0].set_title('Residuals vs Fitted')
    plot_lm_1.axes[0].set_xlabel('Fitted values')
    plot_lm_1.axes[0].set_ylabel('Residuals')


    abs_resid = model_abs_resid.sort_values(ascending=False)
    abs_resid_top_3 = abs_resid[:3]

    for i in abs_resid_top_3.index:
        plot_lm_1.axes[0].annotate(i,
                                   xy=(model_fitted_y[i],
                                       model_residuals[i]));


    QQ = ProbPlot(model_norm_residuals)
    plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

    plot_lm_2.set_figheight(4)
    plot_lm_2.set_figwidth(6)

    plot_lm_2.axes[0].set_title('Normal Q-Q')
    plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
    plot_lm_2.axes[0].set_ylabel('Standardized Residuals');


    abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
    abs_norm_resid_top_3 = abs_norm_resid[:3]

    for r, i in enumerate(abs_norm_resid_top_3):
        plot_lm_2.axes[0].annotate(i,
                                   xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                       model_norm_residuals[i]));


    plot_lm_3 = plt.figure(3)
    plot_lm_3.set_figheight(4)
    plot_lm_3.set_figwidth(6)

    plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
    sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt,
                scatter=False,
                ci=False,
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    plot_lm_3.axes[0].set_title('Scale-Location')
    plot_lm_3.axes[0].set_xlabel('Fitted values')
    plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');


    abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
    abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]

    for i in abs_norm_resid_top_3:
        plot_lm_3.axes[0].annotate(i,
                                   xy=(model_fitted_y[i],
                                       model_norm_residuals_abs_sqrt[i]));

    plot_lm_4 = plt.figure(4)
    plot_lm_4.set_figheight(4)
    plot_lm_4.set_figwidth(6)

    plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
    sns.regplot(model_leverage, model_norm_residuals,
                scatter=False,
                ci=False,
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    plot_lm_4.axes[0].set_xlim(0, 0.20)
    plot_lm_4.axes[0].set_ylim(-3, 5)
    plot_lm_4.axes[0].set_title('Residuals vs Leverage')
    plot_lm_4.axes[0].set_xlabel('Leverage')
    plot_lm_4.axes[0].set_ylabel('Standardized Residuals')


    leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]

    for i in leverage_top_3:
        plot_lm_4.axes[0].annotate(i,
                                   xy=(model_leverage[i],
                                       model_norm_residuals[i]))


    def graph(formula, x_range, label=None):
        x = x_range
        y = formula(x)
        plt.plot(x, y, label=label, lw=1, ls='--', color='red')

    p = len(model_fit.params)

    graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x),
          np.linspace(0.001, 0.200, 50),
          'Cook\'s distance') # 0.5 line

    graph(lambda x: np.sqrt((1 * p * (1 - x)) / x),
          np.linspace(0.001, 0.200, 50)) # 1 line

    plt.legend(loc='upper right');

    plt.show()
    sns.reset_orig()

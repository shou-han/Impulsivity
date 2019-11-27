import pandas as pd
import matplotlib.pyplot as plt
import numpy as npy
# matplotlib inline
import hddm
from patsy import dmatrix


data = hddm.load_csv('data/fmri_modeling.csv')
data = hddm.utils.flip_errors(data)
data = data[data.run >2] # remove counts
data = data[~npy.isnan(data.rt)] #remove nans
fig = plt.figure()
ax = fig.add_subplot(111, xlabel='RT', ylabel='count', title='RT distributions')
for i, subj_data in data.groupby('subj_idx'):
    subj_data.rt.hist(bins=20, histtype='step', ax=ax)

plt.savefig('fMRI_rt.pdf')

#
#
dmatrix("C(stim,Treatment(1))", data.head(10))
m_stim = hddm.HDDM(data, depends_on={'v': 'stim','a':'stim','t':'stim'}, p_outlier=0.05)
m_stim.find_starting_values()
m_stim.sample(100, burn=10)

m_stim.plot_posteriors(['a', 't', 'v', 'a_std'])
m_stim.plot_posterior_predictive(figsize=(18, 5))


v_colour,v_motion = m_stim.nodes_db.node[['a(1)', 'a(2)']]
hddm.analyze.plot_posterior_nodes([v_colour, v_motion])
plt.xlabel('drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior of drift-rate group means')
plt.savefig('fmri_colour_motion.pdf')

## stimulus-coding regression model


#
#
# #stats = m.gen_stats()
# #stats[stats.index.isin(['a', 'a_std', 'a_subj.0', 'a_subj.1'])]
# #m.plot_posteriors(['a', 't', 'v', 'a_std'])
#
# m_within_subj = hddm.HDDMRegressor(data, "v ~ C(stim, Treatment(1))")
# m_within_subj.sample(1000, burn=100)
#
# m_stim.plot_posterior_predictive(figsize=(14, 10))
#
# v_WW, v_LL, v_WL = m_stim.nodes_db.node[['v(WW)', 'v(LL)', 'v(WL)']]
# hddm.analyze.plot_posterior_nodes([v_WW, v_LL, v_WL])
# plt.xlabel('drift-rate')
# plt.ylabel('Posterior probability')
# plt.title('Posterior of drift-rate group means')
# plt.savefig('hddm_demo_fig_06.pdf')
#
# print "P(WW > LL) = ", (v_WW.trace() > v_LL.trace()).mean()
# print "P(LL > WL) = ", (v_LL.trace() > v_WL.trace()).mean()
# print "Stimulus model DIC: %f" % m_stim.dic
#
#
# from patsy import dmatrix
# dmatrix("C(stim, Treatment('WL'))", data.head(50))
# m_within_subj = hddm.HDDMRegressor(data, "v ~ C(stim, Treatment('WL'))")
# m_within_subj.sample(1000,burn=200)
# v_WL, v_LL, v_WW = m_within_subj.nodes_db.ix[["v_Intercept",
#                                               "v_C(stim, Treatment('WL'))[T.LL]",
#                                               "v_C(stim, Treatment('WL'))[T.WW]"], 'node']
# hddm.analyze.plot_posterior_nodes([v_WL, v_LL, v_WW])
# plt.xlabel('drift-rate')
# plt.ylabel('Posterior probability')
# plt.title('Group mean posteriors of within-subject drift-rate effects.')
# plt.savefig('hddm_demo_fig_07.pdf')
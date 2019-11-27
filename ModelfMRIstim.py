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
m_stim = hddm.HDDMStimCoding(data, include='z', stim_col='stim', split_param='z', depends_on={'v': 'response','a':'response','t':'response'}, p_outlier=0.05)
m_stim.find_starting_values()
m_stim.sample(100, burn=5)

m_stim.plot_posteriors(['a', 't', 'v', 'a_std'])
m_stim.plot_posterior_predictive(figsize=(18, 5))

v_correct,v_incorrect = m_stim.nodes_db.node[['v(0)', 'v(1)']]
hddm.analyze.plot_posterior_nodes([v_correct, v_incorrect])
plt.xlabel('drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior of drift-rate group means')
plt.savefig('fmri_colour_motion.pdf')

#modelVars = {'v1':m_stim['v(0)'].trace(),'v2':m_stim['v(1)'].trace()}
modelVars = m_stim.get_traces()
export_csv = pd.DataFrame(modelVars).to_csv(r'C:\Git Folder\modellingfMRI\test11.csv', index = None, header=True)
## stimulus-coding regression model
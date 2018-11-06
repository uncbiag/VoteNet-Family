from evaluate_results import *

## draw bar graph
res_unet_seg = np.load('./results_folder/unet_seg_res.npy')
res_unet_seg_mean = np.mean(res_unet_seg, axis=1)
res_unet_seg_std = np.std(res_unet_seg, axis=1)

res_all_majority_voting = np.load('./results_folder/all_atlases_seg_res.npy')
res_all_majority_voting_mean = np.mean(res_all_majority_voting, axis=1)
res_all_majority_voting_std = np.std(res_all_majority_voting, axis=1)

res_top_6_majority_voting = np.load('./results_folder/voted_global_seg_res.npy')
res_top_6_majority_voting_mean = np.mean(res_top_6_majority_voting, axis=1)
res_top_6_majority_voting_std = np.std(res_top_6_majority_voting, axis=1)

res_top_6_local_majority_voting = np.load('./results_folder/top_15_local_seg_res.npy')
res_top_6_local_majority_voting_mean = np.mean(res_top_6_local_majority_voting, axis=1)
res_top_6_local_majority_voting_std = np.std(res_top_6_local_majority_voting, axis=1)

ind = np.arange(len(res_unet_seg))  # the x locations for the groups
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - 2*width, res_unet_seg_mean, width, yerr=res_unet_seg_std, color='SkyBlue', label='Unet')
rects2 = ax.bar(ind - width, res_all_majority_voting_mean, width, yerr=res_all_majority_voting_std, color='IndianRed', label='All Majority Voting')
rects3 = ax.bar(ind, res_top_6_majority_voting_mean, width, yerr=res_top_6_majority_voting_std, color='Blue', label='Top 6 Majority Voting')
rects4 = ax.bar(ind + width, res_top_6_local_majority_voting_mean, width, yerr=res_top_6_local_majority_voting_std, color='Red', label='Top 6 Local Majority Voting')


ax.set_ylabel('Dice Score')
ax.set_title('Scores by Methods')
ax.set_xticks(ind)
ax.set_xticklabels(('S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40'))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.savefig('test_3.png', dpi=1000)
# File structure for RSA-cross-species-analysis
Creates group-level averages of similarity maps for each

# "P:\userdata\raulh87\data\EmoB\results\RSA\basic-block\by_group\groups\angry\D-sub-01"

-- datafolder/dataset/results/RSA
 -[model]
  -by_class
   -classes: folder, similarity map by class, class average similarity maps for all pairwise combinations in comparisons
    -[specie]_r_[radius]_[method]_[rsa_method]_class_[class]_mean.nii.gz
    -[specie]_r_[radius]_[method]_[rsa_method]_class_[class]_std.nii.gz
    -[specie]_r_[radius]_[method]_[rsa_method]_class_[class]_z.nii.gz
   -comparisons: folder, difference similarity map by class, difference class average similarity maps for all pairwise combinations in comparisons
    -[specie]_r_[radius]_[method]_[rsa_method]_comp_[comparison_model]_mean.nii.gz
    -[specie]_r_[radius]_[method]_[rsa_method]_comp_[comparison_model]_std.nii.gz
    -[specie]_r_[radius]_[method]_[rsa_method]_comp_[comparison_model]_z.nii.gz
-- datafolder/dataset/rsa_models
 -by_class: 
  -class_[class].csv: table with two columns, cat1 and cat2, these form the pairwise combinations of conditions that conform the class of pairs.
  -comp_[comparison_model].txt: text file with two lines, line 1 is class_a, line 2 is class_b, these form the contrast between two pairs of classes. The classes of pairs are defined in the class_[class].csv files. The comparisons are defined in the comp_[comparison_model].txt files. The RSA results will be the difference between the average similarity maps of the two classes of pairs defined in the comparison model.

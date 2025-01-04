1. Familiarizarea cu dataset-ul
2. Parameters freezing pentru primele 5 iteratii
3. Research metode prin care se poate elimita noise-ul (clusterizare pe baza clasei)
4. Folosit augumentare - obtinut un scor mai rau
   1. When rotating, make the black pixels have the mean value of the pixels
   2. When cropping, make the black pixels have the value of the edge pixels
5. Folosit CutMix - creste mult mai greu accuracy, dar macar nu este problema de overfitting (12% train vs 19% val)

Updated
- folosit autoaugment pentru CIFAR10 - improvement cu 40%
- folosit warmup for epocs mai mare
- 40.2% noise in the dataset
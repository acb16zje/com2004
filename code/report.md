# OCR assignment report


## Feature Extraction (Max 200 Words)


Steps:
1. Use the provided `process_training_data` function to extract features from training data
2. Use PCA with 40 axes to reduce the perform dimensionality reduction on the train data
3. Create a list of tuples called `char_compare_list` which contain all possible comparision
   between unique labels, excluding self and duplicate comparision
4. Loop through `char_compare_list` and calculate the divergence of every pair of labels
5. Pick the top 25 features with highest divergence using `np.argsort`
6. Loop through `char_compare_list` again and try to find the best overall 10 features using
   multidivergence in the list of 25 features. The best 10 features of each pair of labels
   are stored in an array.
7. Count the top 10 most common features, and that will be our best features.

Initially I used divergence and correlation to find the top 10 features, but the performance
was not good. In the multidivergence part, it requires a first feature to be provided first in
order to find the best pair. So by trial and error, I have found out that feature 1 gives the
best result.


## Classifier (Max 200 Words)


Originally I used the single nearest neighbour classifer and the result was already impressive.
But the results on pages with high noise level can be improved. So I modified the original
single nearest neighbour classifer to a k-nearest-neighbour classifier and used `k = 3`.
If `k = 1`, then it will become a single nearest neighbour classifier. The classifier will loop
through the test page, and classified the labels based on its `k` amount of neighbours. For each
label in the test page, it looks for its `k` nearest neighbour and return the most common label.
High amount of `k` makes the results on clear pages worse, but it does improve the results on
pages with high noise level. By trial and error, 3 seems to be the optimum value for `k`.


## Error Correction (Max 200 Words)


The test pages in the `dev` folder are using UK English. I have found a word list online which
contains about 100k words sorted according to the word frequency. The word list contain some
French and German words so I have to remove them and add some words manualluy.

Steps:
1. Declare two variables called `start_pos` and `end_pos` for recording the starting and
   ending position of a word respectively.
2. Start by looping through all the labels returned by `classify_page` function. Calculate the
   ending x-coordinate of the current label and the starting x-coordinate of the next label.
   If the distance exceed a certain value, then join the labels starting from `start_pos` to
   `end_pos + 1`.
3. The error correction will create a temporary dictionary for each word. It will then
   look for the most similar word in the temporary dictionary. If the similarity exceeds a
   threshold, then the word will be corrected.
4. After the error correction on the word, set the value of `start_pos` to `end_pos + 1` and
   continue the loop.
5. Each successful loop will increase the value of `end_pos` by 1. The loop continues until
   it reaches the end of the array.


## Performance


The percentage errors (to 1 decimal place) for the development data are as follows:
- Page 1: 97.9%
- Page 2: 99.0%
- Page 3: 97.9%
- Page 4: 85.4%
- Page 5: 67.7%
- Page 6: 54.3%


## Other information (Optional, Max 100 words)


In `process_training_data`, I tried adding some noise into the train data. The results on page 4 to 6 were better.
But there are more performance drop on page 1 to page 3 than the improvement. So I
commented it out.

In `load_test_page`, I tried to remove the noise from the pages. For each page, it the "number of noise"
exceeds a certain threshold, then noise reduction will be performed on that page. The noise reduction
will set pixels that are lower than a certain threshold to 0, and pixels that are higher than a threshold
to 255. 


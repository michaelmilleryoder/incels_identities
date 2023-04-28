library('jsonlite')
library("stm")

# Settings
num_topics <- 30
max_iters <- 100
do_stem <- TRUE
min_df <- 100 # minimum document frequency for words
covariates_str <- 'top5percent_eigen_user'
reprocess <- TRUE # whether to force new preprocessing of documents


cat("Loading corpus...\n")
corpus_name <- 'comments_user_info'
path = '../../data/incels/processed_comments_user_info.jsonl'
data <- stream_in(file(path))


cat('Preparing data for STM...\n')
docs_outpath <- sprintf('../tmp/processed_%s.rds', corpus_name)
customstops = c("’re", "n’t", "’ve", "’ll", "'re", "n't", "'ll", "you're", "don't", "I've", "you've", "I'll",
		 "like", "anything", "something", "will", "come", "get", "got",
		"seem", "even", "still", "though", "one")
if (file.exists(docs_outpath) & !reprocess) {
	out <- readRDS(docs_outpath)
	cat(sprintf('\n\tLoaded processed docs from %s\n', docs_outpath))
} else {
	processed <- textProcessor(data$content, stem=do_stem, metadata=data,
			removestopwords=TRUE, customstopwords=customstops,
			verbose=TRUE)
	out <- prepDocuments(processed$documents, processed$vocab, processed$meta,
						lower.thresh=min_df)

	# Save out processed docs for quicker loading
	saveRDS(out, docs_outpath)
	cat(sprintf('\n\tSaved processed docs to %s\n', docs_outpath))
}

cat('\nEstimating STM...\n')
estimated <- stm(documents = out$documents,
                vocab = out$vocab,
                K = num_topics,
                max.em.its = max_iters,
                prevalence =~ top5percent_eigen_user,
		verbose=TRUE,
                data = out$meta)

outpath <- sprintf('../models/%s_stm_%s_%dtopics_%dit_%dmindf.rds', corpus_name, covariates_str, num_topics, max_iters, min_df)
cat(sprintf('Saved model to %s\n', outpath))
saveRDS(estimated, outpath)

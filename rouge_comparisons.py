from rouge_score import rouge_scorer

# compare our summary with gpt_4o_mini on the query "What causes climate change?"
our_summary = "We've sorted your query into the 'climate_impacts' category. The fully modified query is: 'What is the cause of climate change?'. Find our answer here: Based on the three most relevant documents in our database (Measuring the economic impact of climate change on crop production in the dry zone of Myanmar: a Ricardian approach by Aung Tun Oo et al., 2020; The demographic implications of climate change for Aotearoa New Zealand: A review by Cameron, 2013; and The poverty impacts of climate change: a review of the evidence by Skoufias et al., 2011), climate change is primarily driven by the accumulation of greenhouse gases in the atmosphere from human activities such as burning fossil fuels, deforestation, and industrial processes. These emissions trap heat, leading to global temperature rises that alter precipitation patterns, intensify extreme weather events, and disrupt ecosystems. Over time, these changes to the climate system are exacerbated by feedback mechanisms, resulting in adverse impacts on agriculture, demographics, and poverty levels worldwide."
gpt_4o_summary = "Climate change is primarily caused by human activities that increase the concentration of greenhouse gases in the atmosphere, leading to a rise in global temperatures. The burning of fossil fuels such as coal, oil, and natural gas for energy and transportation is the largest contributor, releasing large amounts of carbon dioxide (CO₂) into the air. Deforestation also plays a significant role, as trees that absorb CO₂ are cut down, reducing the planet’s ability to regulate carbon levels. Additionally, industrial processes, agriculture, and waste management produce other potent greenhouse gases like methane (CH₄) and nitrous oxide (N₂O). While natural factors such as volcanic activity and variations in solar radiation can influence the climate, the rapid and large-scale changes observed in recent decades are overwhelmingly due to human influence."

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

scores = scorer.score(our_summary, gpt_4o_summary)

# Output
for key, score in scores.items():
    print(f"{key}: P={score.precision:.2f}, R={score.recall:.2f}, F={score.fmeasure:.2f}")

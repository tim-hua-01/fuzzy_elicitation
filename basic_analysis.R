setwd("~/Documents/aisafety_githubs/philosophy_explore")
if (!require(tidyverse)) install.packages("tidyverse"); library(tidyverse)
if (!require(magrittr)) install.packages("magrittr"); library(magrittr)
library(jsonlite)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(skimr)
library(modelsummary)
library(fixest)
library(ggtext)
library(broom)
library(ks)
library(ggpattern)
library(readr)
myTheme <- theme(plot.title = element_text(size = 15),
                 panel.background = element_rect(fill = '#F2F2ED'),
                 legend.text = element_text(size = 12),
                 legend.title = element_text(size = 12),
                 plot.subtitle = element_text(size = 12),
                 axis.title = element_text(size = 12),
                 strip.text = element_text(size = 12),
                 axis.text = element_text(size = 12, colour = 'black'),
                 legend.position = "bottom",
                 legend.background = element_rect(linetype = 3,size = 0.5, color = 'black', fill = 'grey94'),
                 legend.key = element_rect(size = 0.5, linetype = 1, color = 'black'))

#I also have some nice colors that I use in my various graphs.
nicepurp <- "#6C5078"
niceblue <- '#38A5E0'
nicegreen <- '#A3DCC0'

custom_colors <- c("#2ECC71", "#A3E635", "#F4D03F", "#F39C12", "#E74C3C", "#C0392B", "#0072B2", "#CC79A7")


ggsaver <- function(filename, sc = 1) {
  ggsave(filename = paste0("iclr2026/", filename, ".png"), width = 6, height = 4, scale = sc)
}


df <- read_csv("philosophy_explore/data/v2/all_results.csv") %>%
  mutate(prompt_variant = ifelse(!is.na(prompt_variant), prompt_variant, "human")) %>% filter(!is.na(answer))
df %<>% mutate(normaizled_score = (total - mean(total))/sd(total),
               normalized_reasoning_char = (reasoning_char_count - mean(reasoning_char_count))/
                                              sd(reasoning_char_count),
               normalized_answer_char = (answer_char_count - mean(answer_char_count))/
                                              sd(answer_char_count))

o3 <- read_csv("data/v2_o3/all_results.csv")
o3 %>% skim(total)
df %>% skim(total, claude_total)
o3 %<>% mutate(normalize_score = (total - mean(total))/sd(total))

feols(normalize_score ~ as.factor(prompt_variant) - 1,
      data = o3, vcov = ~question_id)

m1 <- feols(normalize_score ~ as.factor(prompt_variant),
            data = o3, vcov = ~question_id)

m2 <- feols(normalize_score ~ as.factor(prompt_variant) +
              output_tokens +
              reasoning_tokens,
            data = o3, vcov = ~question_id)

m3 <- feols(normalize_score ~ as.factor(prompt_variant) +
              output_tokens +
              reasoning_tokens +
              output_tokens*reasoning_tokens +
              I(output_tokens^2) +
              I(reasoning_tokens^2),
            data = o3, vcov = ~question_id)

modelsummary(
  list("Model 1" = m1, "Model 2" = m2, "Model 3" = m3),
  statistic = "conf.int",
  stars = TRUE,
  gof_omit = "IC|Log|FE|RMSE"
)



claude_grades <- read_csv("data/v2/all_results_anthropic_claude-sonnet-4-5.csv") %>%
  mutate(prompt_variant = ifelse(!is.na(prompt_variant), prompt_variant, "human")) %>% filter(!is.na(answer))

claude_grades %<>% mutate(claude_total = total, claude_normalized = (total - mean(total))/sd(total),
                          normalized_reasoning_char = (reasoning_char_count - mean(reasoning_char_count))/
                            sd(reasoning_char_count),
                          normalized_answer_char = (answer_char_count - mean(answer_char_count))/
                            sd(answer_char_count))

claude_grades%>% select(total) %>% skim()

df %<>% left_join(claude_grades %>% select(claude_total, claude_normalized, question_id, sample_idx,prompt_variant))


df %>% skim(total)

df %>% group_by(prompt_variant) %>% skim()

df %>% group_by(prompt_variant, question_id) %>% summarize(best_of_n = max(total), medianofn = median(total),
                                                           worst = min(total))%>% 
  group_by(prompt_variant) %>% skim()

df %>% filter(total <= 30 | total >= 33) %>% group_by(total > 30) %>% count(prompt_variant)

feols(total ~ as.factor(prompt_variant)+ 
        normalized_answer_char + normalized_reasoning_char, data = df, vcov = ~question_id)
feols(total ~ as.factor(prompt_variant), data = df, vcov = ~question_id)
feols(claude_total ~ as.factor(prompt_variant), data = df, vcov = ~question_id)

feols(claude_normalized ~ as.factor(prompt_variant), data = df, vcov = ~question_id)

feols(claude_total ~ as.factor(prompt_variant)+ 
        normalized_answer_char + normalized_reasoning_char, data = df, vcov = ~question_id)

model <- feols(normaizled_score ~ as.factor(prompt_variant), data = df, vcov = ~question_id)

# Extract coefficients from model using tidy
model_coefs <- tidy(model, conf.int = TRUE)

# Create data frame for plotting
# Baseline (answer_with_rubric_2k) is the intercept, normalized to 0
regression_data <- data.frame(
  prompt_variant = c("answer_with_rubric_2k", 
                     "answer_without_rubric_2k", 
                     "rubric_v2_tryhard"),
  estimate = c(0, 
               model_coefs$estimate[model_coefs$term == "as.factor(prompt_variant)answer_without_rubric_2k"],
               model_coefs$estimate[model_coefs$term == "as.factor(prompt_variant)rubric_v2_tryhard"]),
  std_error = c(0,#model_coefs$std.error[model_coefs$term == "(Intercept)"],
                model_coefs$std.error[model_coefs$term == "as.factor(prompt_variant)answer_without_rubric_2k"],
                model_coefs$std.error[model_coefs$term == "as.factor(prompt_variant)rubric_v2_tryhard"])
)

# Create bar chart with error bars (deviations from baseline)
ggplot(regression_data, aes(x = prompt_variant, y = estimate, fill = prompt_variant)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_errorbar(aes(ymin = estimate - 1.96 * std_error, 
                    ymax = estimate + 1.96 * std_error),
                width = 0.3, linewidth = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
  scale_fill_manual(values = c("answer_with_rubric_2k" = niceblue,
                               "answer_without_rubric_2k" = "#E74C3C",
                               "rubric_v2_tryhard" = nicegreen)) +
  labs(title = "Effect of Prompt Variant on Normalized Score",
       subtitle = "Baseline: answer_with_rubric_2k (normalized to 0), Error bars: 95% CI",
       x = "Prompt Variant",
       y = "Deviation from Baseline",
       fill = "Prompt Variant") +
  myTheme +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Version with absolute scores (no intercept model)
model_no_intercept <- feols(normaizled_score ~ as.factor(prompt_variant) - 1+ 
                              normalized_answer_char + normalized_reasoning_char, data = df, vcov = ~question_id)

# Extract coefficients from no-intercept model
model_coefs_no_int <- tidy(model_no_intercept, conf.int = TRUE) %>% filter(str_detect(term, "as.fac"))

# Clean up term names and create data frame
regression_data_absolute <- model_coefs_no_int %>%
  mutate(prompt_variant = str_replace(term, "as.factor\\(prompt_variant\\)", "")) %>%
  select(prompt_variant, estimate, std_error = std.error)

# Create bar chart with absolute scores
regression_data_absolute %>% 
  mutate(prompt_variant = case_when(
    prompt_variant == 'answer_with_rubric_2k' ~ "With Rubric",
    prompt_variant == 'answer_without_rubric_2k' ~ "No Rubric",
    prompt_variant == 'rubric_v2_tryhard' ~ "With Rubric & Try Hard",
    TRUE ~ "No Rubric & Try Hard"
  )) %>%
  ggplot(aes(x = prompt_variant, y = estimate, fill = prompt_variant)) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_errorbar(aes(ymin = estimate - 1.96 * std_error, 
                    ymax = estimate + 1.96 * std_error),
                width = 0.2, linewidth = 0.5) +
  scale_fill_manual(values = c(niceblue,"#E74C3C",nicegreen, nicepurp)) +
  scale_y_continuous(breaks = seq(-1,1,0.25)) +
  labs(title = "Prompting the GLM-4.7 to try harder increases self-graded score 
on philosophy questions",
       subtitle = "95% confident intervals shown. Score normalized to have zero mean and unit standard deviation.",
       x = "Prompt Variant",
       y = "Normalized Score",
       fill = "Prompt Variant",
       caption = "Ten philosophy questions sampled 30 times each. Standard errors clustered at the question level
Controlling for response and reasoning lengths.") +
  myTheme + theme(legend.position = "none")

# Bar chart comparing total vs claude_total across prompt variants (no intercept)
model_total <- feols(total ~ as.factor(prompt_variant) - 1, data = df, vcov = ~question_id)
model_claude <- feols(claude_total ~ as.factor(prompt_variant) - 1, data = df, vcov = ~question_id)

# Extract coefficients
coefs_total <- tidy(model_total, conf.int = TRUE)
coefs_claude <- tidy(model_claude, conf.int = TRUE)

# Create combined data frame with both grader types
combined_data <- bind_rows(
  coefs_total %>%
    mutate(prompt_variant = str_replace(term, "as.factor\\(prompt_variant\\)", "")) %>%
    select(prompt_variant, estimate, std_error = std.error) %>%
    mutate(grader = "Self Grade"),
  coefs_claude %>%
    mutate(prompt_variant = str_replace(term, "as.factor\\(prompt_variant\\)", "")) %>%
    select(prompt_variant, estimate, std_error = std.error) %>%
    mutate(grader = "Claude Grade")
)

# Create side-by-side bar chart
combined_data %>%
  mutate(prompt_variant = case_when(
    prompt_variant == 'answer_with_rubric_2k' ~ "With Rubric",
    prompt_variant == 'answer_without_rubric_2k' ~ "No Rubric",
    prompt_variant == 'rubric_v2_tryhard' ~ "With Rubric & Try Hard",
    TRUE ~ "No Rubric & Try Hard"
  )) %>%
  ggplot(aes(x = prompt_variant, y = estimate, fill = grader)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = estimate - 1.96 * std_error, 
                    ymax = estimate + 1.96 * std_error),
                position = position_dodge(width = 0.8),
                width = 0.3, linewidth = 0.5) +
  coord_cartesian(ylim = c(28,36)) + 
  scale_fill_manual(values = c("#D4A574","grey40")) +
  labs(title = "Claude graded scores follow a similar pattern—trying hard prompt yields higher scores
compared to rubric only by 0.37 standard deviations",
       subtitle = "95% confidence intervals shown. Standard errors clustered at question level.",
       x = "Prompt Variant",
       y = "Raw Rubric Score out of 48",
       fill = "Grader Type") +
  myTheme



feols(normaizled_score ~ as.factor(prompt_variant) + 
        normalized_answer_char + normalized_reasoning_char, data = df, vcov = ~question_id)

feols(normalized_answer_char ~ as.factor(prompt_variant) , data = df, vcov = ~question_id)

feols(normalized_reasoning_char ~ as.factor(prompt_variant) , data = df, vcov = ~question_id)

df %>% mutate(question_id_short = str_sub(question_id, 1, 17)) %>%
  group_by(question_id_short, prompt_variant) %>% 
  summarise(mean_tot = mean(total), sd_tot = sd(total), 
            max_tot = max(total)) %>%
  ggplot(aes(prompt_variant, mean_tot)) + 
  geom_col(width = 0.7) + facet_wrap(~question_id_short) + 
  myTheme  + coord_flip(ylim = c(29,36))


df %>% mutate(question_id_short = str_sub(question_id, 1, 17)) %>%
  group_by(question_id_short, prompt_variant) %>% 
  summarise(mean_tot = mean(total), sd_tot = sd(total)) %>%
  ggplot(aes(prompt_variant, sd_tot)) + 
  geom_col(width = 0.7) + facet_wrap(~question_id_short) + 
  myTheme  + coord_flip(ylim = c(1,3))

df %>% mutate(question_id_short = str_sub(question_id, 1, 17)) %>%
  group_by(question_id_short, prompt_variant) %>% 
  summarise(mean_tot = mean(total), sd_tot = sd(total), 
            p90 = quantile(total,0.9)) %>%
  ggplot(aes(prompt_variant, p90)) + 
  geom_col(width = 0.7) + facet_wrap(~question_id_short) + 
  myTheme  + coord_flip(ylim = c(30,40))

library(fixest)
library(modelsummary)

m1 <- feols(normaizled_score ~ as.factor(prompt_variant),
            data = df, vcov = ~question_id)

m2 <- feols(normaizled_score ~ as.factor(prompt_variant) +
              normalized_answer_char +
              normalized_reasoning_char,
            data = df, vcov = ~question_id)

m3 <- feols(normaizled_score ~ as.factor(prompt_variant) +
              normalized_answer_char +
              normalized_reasoning_char +
              normalized_answer_char*normalized_reasoning_char +
              I(normalized_answer_char^2) +
              I(normalized_reasoning_char^2),
            data = df, vcov = ~question_id)

modelsummary(
  list("Model 1" = m1, "Model 2" = m2, "Model 3" = m3),
  statistic = "conf.int",
  stars = TRUE,
  gof_omit = "IC|Log|FE|RMSE"
)

m1 <- feols(claude_normalized ~ as.factor(prompt_variant),
            data = df, vcov = ~question_id)

m2 <- feols(claude_normalized ~ as.factor(prompt_variant) +
              normalized_answer_char +
              normalized_reasoning_char,
            data = df, vcov = ~question_id)

m3 <- feols(claude_normalized ~ as.factor(prompt_variant) +
              normalized_answer_char +
              normalized_reasoning_char +
              normalized_answer_char*normalized_reasoning_char +
              I(normalized_answer_char^2) +
              I(normalized_reasoning_char^2),
            data = df, vcov = ~question_id)

modelsummary(
  list("Model 1" = m1, "Model 2" = m2, "Model 3" = m3),
  statistic = "conf.int",
  stars = TRUE,
  gof_omit = "IC|Log|FE|RMSE"
)

# Bar chart for o3 data (no intercept model)
model_o3 <- feols(normalize_score ~ as.factor(prompt_variant) - 1,
                  data = o3, vcov = ~question_id)

# Extract coefficients
coefs_o3 <- tidy(model_o3, conf.int = TRUE)

# Create data frame for plotting
o3_data <- coefs_o3 %>%
  mutate(prompt_variant = str_replace(term, "as.factor\\(prompt_variant\\)", "")) %>%
  select(prompt_variant, estimate, std_error = std.error)

# Create bar chart
o3_data %>%
  mutate(prompt_variant = case_when(
    prompt_variant == 'answer_with_rubric_2k' ~ "With Rubric",
    prompt_variant == 'answer_without_rubric_2k' ~ "No Rubric",
    prompt_variant == 'rubric_v2_tryhard' ~ "With Rubric & Try Hard",
    TRUE ~ "No Rubric & Try Hard"
  )) %>%
  ggplot(aes(x = prompt_variant, y = estimate, fill = prompt_variant)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_errorbar(aes(ymin = estimate - 1.96 * std_error, 
                    ymax = estimate + 1.96 * std_error),
                width = 0.3, linewidth = 0.5) +
  scale_fill_manual(values = c(niceblue, "#E74C3C", nicegreen, nicepurp)) +
  scale_y_continuous(breaks = seq(-1,1,0.25)) + 
  labs(title = "o3 Model: Normalized Scores by Prompt Variant",
       subtitle = "95% confidence intervals shown. Standard errors clustered at question level.",
       x = "Prompt Variant",
       y = "Normalized Score",
       fill = "Prompt Variant") +
  myTheme + theme(legend.position = "none")



p <- 15 / 365
n <- 30

p0 <- (1 - p)^n
p1 <- choose(n, 1) * p * (1 - p)^(n - 1)

prob <- 1 - p0 - p1

cat("P(X = 0):", p0, "\n")
cat("P(X = 1):", p1, "\n")
cat("P(X >= 2):", prob, "\n")


# ============================================================
# PAIRWISE COMPARISON ANALYSIS
# Does prompt format predict pairwise outcomes beyond score?
# ============================================================

pw <- read_csv("data/v2/stratified_real.csv")
pwr <- read_csv("data/v2/stratified_wrong_ins.csv")
# Map prompt variants to binary features:
#   answer_with_rubric_2k    -> rubric=1, tryhard=0
#   answer_without_rubric_2k -> rubric=0, tryhard=0
#   rubric_v2_tryhard        -> rubric=1, tryhard=1
#   v2_tryhard_norubric      -> rubric=0, tryhard=1
rubric_variants  <- c("answer_with_rubric_2k", "rubric_v2_tryhard")
tryhard_variants <- c("rubric_v2_tryhard", "v2_tryhard_norubric")

pw %<>% mutate(
  high_has_rubric  = high_prompt_variant %in% rubric_variants,
  low_has_rubric   = low_prompt_variant  %in% rubric_variants,
  high_has_tryhard = high_prompt_variant %in% tryhard_variants,
  low_has_tryhard  = low_prompt_variant  %in% tryhard_variants
)

pwr %<>% mutate(
  high_has_rubric  = high_prompt_variant %in% rubric_variants,
  low_has_rubric   = low_prompt_variant  %in% rubric_variants,
  high_has_tryhard = high_prompt_variant %in% tryhard_variants,
  low_has_tryhard  = low_prompt_variant  %in% tryhard_variants
)

feols(model_picked_high == "True" ~ 1, 
      data = pw %>% filter(model_picked_high != 'disagree'
                           ), weights = ~weight, vcov = ~question_id)

feols(model_picked_high == "True" ~ as.factor(high_prompt_variant), 
      data = pw %>% filter(model_picked_high != 'disagree'
      ), weights = ~weight, vcov = ~question_id)

feols(model_picked_high == "True" ~ as.factor(low_prompt_variant), 
      data = pw %>% filter(model_picked_high != 'disagree'
      ), weights = ~weight, vcov = ~question_id)

# Only keep pairs where both orderings agreed
pw_agreed <- pw %>% filter(model_picked_high != "disagree")

# --- RUBRIC ANALYSIS ---
# Subset: exactly one entry has a rubric, the other doesn't
# Recode so A = rubric entry, B = non-rubric entry
pw_rubric <- pw_agreed %>%
  filter(high_has_rubric != low_has_rubric) %>%
  mutate(
    # A = the entry with rubric, B = the entry without
    a_score = ifelse(high_has_rubric, high_total, low_total),
    b_score = ifelse(high_has_rubric, low_total, high_total),
    score_diff = a_score - b_score,  # positive = rubric entry scored higher
    a_wins = as.numeric(
      (high_has_rubric & model_picked_high == "True") |
      (low_has_rubric  & model_picked_high == "False")
    )
  )

# --- TRYHARD ANALYSIS ---
# Subset: exactly one entry has tryhard, the other doesn't
# Recode so A = tryhard entry, B = non-tryhard entry
pw_tryhard <- pw_agreed %>%
  filter(high_has_tryhard != low_has_tryhard) %>%
  mutate(
    a_score = ifelse(high_has_tryhard, high_total, low_total),
    b_score = ifelse(high_has_tryhard, low_total, high_total),
    score_diff = a_score - b_score,  # positive = tryhard entry scored higher
    a_wins = as.numeric(
      (high_has_tryhard & model_picked_high == "True") |
      (low_has_tryhard  & model_picked_high == "False")
    )
  )

cat("\n=== Rubric subset (A=rubric, B=no rubric) ===\n")
cat("N pairs:", nrow(pw_rubric), "\n")
cat("P(rubric wins):", mean(pw_rubric$a_wins), "\n")
cat("Score diff (A-B) range:", min(pw_rubric$score_diff), "to", max(pw_rubric$score_diff), "\n")

cat("\n=== Tryhard subset (A=tryhard, B=no tryhard) ===\n")
cat("N pairs:", nrow(pw_tryhard), "\n")
cat("P(tryhard wins):", mean(pw_tryhard$a_wins), "\n")
cat("Score diff (A-B) range:", min(pw_tryhard$score_diff), "to", max(pw_tryhard$score_diff), "\n")

# LPM: P(A wins) ~ score_diff + score_diff^2
# Intercept = P(feature-holder wins) when scores are tied
# score_diff = how much each point of A's score advantage increases win prob

rubric_m1 <- feols(a_wins ~ score_diff + I(score_diff^2),
                   data = pw_rubric, vcov = ~question_id)

tryhard_m1 <- feols(a_wins ~ score_diff + I(score_diff^2),
                    data = pw_tryhard, vcov = ~question_id)

modelsummary(
  list("P(rubric wins)" = rubric_m1, "P(tryhard wins)" = tryhard_m1),
  statistic = "conf.int",
  stars = TRUE,
  gof_omit = "IC|Log|FE|RMSE",
  title = "LPM: P(feature-holder wins) controlling for score gap (A-B)"
)

# Interpretation of the INTERCEPT (the key number):
#   > 0.5: feature genuinely improves quality beyond what the score captures
#   < 0.5: feature inflates scores without improving real quality
#   = 0.5: feature effect is fully captured by the rubric score

pw %>% filter(model_picked_high != 'disagree') %>% filter(high_prompt_variant == 'rubric_v2_tryhard' & 
                low_prompt_variant == 'answer_with_rubric_2k') %>% 
  summarise(mean(model_picked_high == "True"), n())
  

pw %>% filter(model_picked_high != 'disagree') %>% filter(low_prompt_variant == 'rubric_v2_tryhard' & 
                                                             high_prompt_variant == 'answer_with_rubric_2k') %>% 
  summarise(mean(model_picked_high == "True"), n())



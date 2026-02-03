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

df %>% skim()

df %>% group_by(prompt_variant) %>% skim()

df %>% group_by(prompt_variant, question_id) %>% summarize(best_of_n = max(total), medianofn = median(total),
                                                           worst = min(total))%>% 
  group_by(prompt_variant) %>% skim()


feols(total ~ as.factor(prompt_variant), data = df, vcov = ~question_id)

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
  labs(title = "Prompting the model to try harder increases self-graded score by 0.5
standard deviations on philosophy questions",
       subtitle = "95% confident intervals shown. Score normalized to have zero mean and unit standard deviation.",
       x = "Prompt Variant",
       y = "Normalized Score",
       fill = "Prompt Variant",
       caption = "Ten philosophy questions sampled 30 times each. Standard errors clustered at the question level
Controlling for response and reasoning lengths.") +
  myTheme#

feols(normaizled_score ~ as.factor(prompt_variant) + 
        normalized_answer_char + normalized_reasoning_char, data = df, vcov = ~question_id)

feols(answer_char_count ~ as.factor(prompt_variant) , data = df, vcov = ~question_id)

feols(reasoning_char_count ~ as.factor(prompt_variant) , data = df, vcov = ~question_id)


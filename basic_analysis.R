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

df %>% group_by(prompt_variant) %>% skim()

df %>% group_by(prompt_variant, question_id) %>% summarize(best_of_n = max(total), medianofn = median(total),
                                                           worst = min(total))%>% 
  group_by(prompt_variant) %>% skim()

df %<>% mutate(normaizled_score = (total - mean(total))/sd(total))

feols(total ~ as.factor(prompt_variant), data = df, vcov = ~question_id)

feols(normaizled_score ~ as.factor(prompt_variant), data = df, vcov = ~question_id)



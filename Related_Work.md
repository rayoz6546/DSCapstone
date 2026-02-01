# 1. GPTs are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models 
 
Eloundou et al. (2023) provide the first systematic assessment of LLM exposure across the U.S. labor market. Using a rubric-based framework combining human expert evaluation with GPT-4 self-assessment, they evaluated 1,016 occupations to determine which work tasks could be significantly affected by LLMs. 

**Key Contributions**

_Methodology_: The authors developed a two-tier exposure framework measuring (α) direct LLM exposure through current interfaces, and (β) LLM + complementary software exposure. Tasks were classified based on whether LLMs could reduce completion time by at least 50%. 

**Findings:**

- 80% of U.S. workers have ≥10% of tasks affected by LLMs; 19% have ≥50% affected 
- Higher-wage occupations show greater exposure, reversing historical automation patterns 
- Software multiplier effect: β exposure (47-56% of tasks) dramatically exceeds α exposure (15%) 
- Provides occupation-level exposure scores for all O*NET SOC codes (publicly available at https://github.com/openai/GPTs-are-GPTs) 

**Limitations and Gaps We Address**

1. While foundational, this work has key limitations our project addresses: 
2. Static Occupation Codes: Provides pre-computed scores for standardized occupations only. Our approach: Dynamic pipeline accepting arbitrary job descriptions, enabling assessment of emerging roles. 
3. Limited Skill Granularity: Evaluates tasks but doesn't extract underlying skills. Our approach: Explicit skill extraction to identify which competencies drive automation risk, enabling skill-level insights. 
4. U.S.-Centric: Confined to U.S. labor market using O*NET. Our approach: Global framework incorporating international taxonomies (ISCO, ESCO) and regional labor market trends. 
5. No Personalization: Occupation-level scores miss within-occupation variation. Our approach: Job-posting-level analysis capturing role-specific task compositions. 

**How We Build on This Work** 

We use Eloundou et al.'s exposure scores as validated ground truth features in our ML model, extending their work by: (1) processing natural language job descriptions rather than requiring occupation codes, (2) extracting and weighting specific skills, (3) incorporating additional contextual features (geography, industry, temporal trends), and (4) enabling global applicability through multi-taxonomy support. Our project transforms their academic insights into a practical, real-time assessment tool for job automation risk. 

**References**

Eloundou, T., Manning, S., Mishkin, P., & Rock, D. (2024). GPTs are GPTs: Labor market impact potential of LLMs. Science, 384(6703), 1306-1308.
Paper: https://arxiv.org/abs/2303.10130
Code: https://github.com/openai/GPTs-are-GPTs

# 2. Measuring the Exposure of Occupations to Artificial Intelligence

Felten et al. (2021) propose an influential framework for measuring how exposed different occupations are to advances in artificial intelligence. Rather than predicting job automation directly, the paper focuses on identifying which occupations rely most heavily on tasks and abilities where AI progress is rapid.

**Key Contributions**

_Methodology:_ The authors construct an AI Occupational Exposure (AIOE) index by linking AI capabilities to detailed task and ability descriptions from O*NET. Exposure is defined as the degree to which an occupation’s tasks align with areas of active AI development.

**Findings**

- AI exposure varies substantially across occupations
- Higher-wage and higher-skill jobs often exhibit greater AI exposure
- Exposure reflects task composition rather than routine/manual intensity

**Outputs**

A reproducible occupation-level exposure measure covering U.S. SOC codes.

**Limitations and Gaps We Address**

1. The AIOE index measures relevance of AI, not automation risk or job displacement.
2. Exposure is computed for standardized occupations rather than role-specific job descriptions.
3. The framework relies on O*NET and U.S. labor market data

**How We Build on This Work**

We view Felten et al.’s exposure index as a complementary perspective on how AI relates to work. Our project builds on this line of research by moving toward job-level analysis from natural language descriptions and by integrating AI exposure concepts into a broader automation-risk assessment framework.

**References**

Felten, E., Raj, M., & Seamans, R. (2021). Measuring the exposure of occupations to artificial intelligence. Strategic Management Journal.
Paper: https://sms.onlinelibrary.wiley.com/doi/epdf/10.1002/smj.3286 
Code: https://github.com/AIOE-Data/AIOE 


# 3. The risk of Automation for Jobs in OECD Countries

Arntz, Gregory, and Zierahn (2016) introduce a task-based framework for estimating automation risk that directly addresses limitations of earlier occupation-level approaches. Although published prior to the recent wave of AI and LLM advances, this work remains foundational because it establishes the methodological principle that automation risk depends on task composition within jobs, rather than job titles alone.

**Key Contributions**

_Methodology:_ Using individual-level data from the OECD’s PIAAC survey, the authors model automation risk based on the types and frequencies of tasks performed by workers. A machine learning classifier estimates the probability that a given task bundle is automatable, and these probabilities are aggregated to produce occupation- and country-level risk estimates.

**Findings**

- Automation risk estimates are substantially lower than earlier occupation-based studies (approximately 9% on average)
- Significant heterogeneity exists within occupations due to differences in task composition
- Cross-country differences in risk are driven largely by how work is organized rather than by technology availability alone

**Why This Paper Remains Relevant**

Despite predating modern AI and LLM systems, this work remains highly relevant because it defines a measurement framework rather than a technology-specific model. Many recent studies—including AI and LLM exposure analyses—implicitly adopt its core insight that job impact depends on task structure. As such, it provides the conceptual foundation on which newer AI-focused work builds.

**Limitations and Gaps We Address**

1. Relies on structured survey responses rather than real-world job descriptions.
2. Task categories are fixed and slow to adapt to emerging roles or evolving skill requirements.
3. Does not account for capabilities unique to modern AI or LLM-based systems.

**How We Build on This Work**

We adopt the task-based perspective introduced by Arntz et al. while extending it to contemporary AI contexts. Our project infers task and skill composition directly from natural language job descriptions and integrates AI-specific exposure signals into a unified automation-risk assessment framework.

**References**

Arntz, M., Gregory, T., & Zierahn, U. (2016). The risk of automation for jobs in OECD countries. OECD Social, Employment and Migration Working Papers, No. 189.
Paper: https://www.oecd.org/content/dam/oecd/en/publications/reports/2016/05/the-risk-of-automation-for-jobs-in-oecd-countries_g17a27d8/5jlz9h56dvq7-en.pdf 


# 4. Artificial Intelligence and Jobs: Evidence from Online Vacancies

Acemoglu, Autor, Hazell, and Restrepo (2022) examine how firms’ adoption of AI technologies relates to changes in hiring patterns and skill demands using establishment-level data on online job vacancies in the United States. Instead of focusing on occupations as static units, this paper tracks real hiring behavior over time to understand how AI exposure shapes labor demand and job requirements.

**Key Contributions**

_Methodology:_ The authors use a comprehensive dataset of online job vacancy postings from 2010 onward, classifying establishments as “AI exposed” when their workforce engages in tasks compatible with current AI capabilities. They analyze how these firms’ vacancy patterns evolve compared to less-exposed establishments, particularly in terms of hiring and skill requirements.

**Findings**

- Rapid growth in AI-related vacancies from 2010 to 2018, particularly in firms already engaged with AI-relevant tasks.
- AI-exposed establishments reduce postings for some traditional skills even as they expand demand for AI-related skills.
- At the aggregate level, the study finds limited detectable impacts of AI exposure on employment and wages, suggesting substitution effects are present but not yet dominant at broader occupational or industry scales.

**Why This Paper Remains Relevant**

Although the paper predates the most recent surge in generative LLM capabilities, it is compelling for related work because it uses actual labor market data to show how AI adoption affects job posting behavior and skill requirements. Its empirical grounding makes it a useful benchmark for understanding how technological change manifests in hiring patterns — an important complement to exposure and task-based studies.

**Limitations and Gaps We Address**

1. The analysis is at the firm/establishment level, not tailored to specific job postings or individual worker profiles.
2. The data timeline precedes widespread adoption of modern LLMs and generative AI.
3. Focuses on hiring patterns and skill demand rather than producing explicit automation-risk probabilities.

**How We Build on This Work**

We incorporate this paper’s empirical insight that AI adoption is observable in changes to job postings and skill requirements. Our project extends this by inferring job-level task and skill structure from natural language and integrating such signals into a predictive automation-risk model that also accounts for AI and LLM exposure.

**References**

Acemoglu, D., Autor, D., Hazell, J., & Restrepo, P. (2022). Artificial intelligence and jobs: Evidence from online vacancies. Journal of Labor Economics, 40(S1), S293–S340.
Paper: https://shapingwork.mit.edu/wp-content/uploads/2023/10/Paper_Artificial-Intelligence-and-Jobs-Evidence-from-Online-Vacancies.pdf 
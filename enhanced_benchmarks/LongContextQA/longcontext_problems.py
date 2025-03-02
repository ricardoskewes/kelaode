"""
Long-context QA problems for language efficiency testing.

This module contains a collection of long-context question answering problems
designed to test language efficiency in processing and reasoning over long texts.
"""

import os
import json
from typing import List, Dict, Any, Optional

# Define the structure of a long-context QA problem
class LongContextQAProblem:
    """
    A long-context question answering problem.
    
    Attributes:
        id: Unique identifier for the problem
        context: Long text context for the problem
        question: Question to be answered based on the context
        answer: Expected answer to the question
        source: Source of the problem (e.g., dataset name)
        difficulty: Difficulty level of the problem
        category: Category of the problem (e.g., summarization, factual QA)
        context_length: Length of the context in characters
    """
    
    def __init__(
        self,
        id: str,
        context: str,
        question: str,
        answer: str,
        source: str = "custom",
        difficulty: str = "medium",
        category: str = "factual_qa",
    ):
        self.id = id
        self.context = context
        self.question = question
        self.answer = answer
        self.source = source
        self.difficulty = difficulty
        self.category = category
        self.context_length = len(context)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the problem to a dictionary."""
        return {
            "id": self.id,
            "context": self.context,
            "question": self.question,
            "answer": self.answer,
            "source": self.source,
            "difficulty": self.difficulty,
            "category": self.category,
            "context_length": self.context_length,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LongContextQAProblem":
        """Create a problem from a dictionary."""
        return cls(
            id=data["id"],
            context=data["context"],
            question=data["question"],
            answer=data["answer"],
            source=data.get("source", "custom"),
            difficulty=data.get("difficulty", "medium"),
            category=data.get("category", "factual_qa"),
        )

# Sample long-context QA problems
SAMPLE_LONGCONTEXT_PROBLEMS = [
    LongContextQAProblem(
        id="longqa_001",
        context="""
The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.

The field of AI research was founded at a workshop held on the campus of Dartmouth College, USA during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true.

Eventually, it became obvious that they had grossly underestimated the difficulty of the project. In 1973, in response to the criticism from James Lighthill and ongoing pressure from the US Congress to fund more productive projects, both the U.S. and British governments cut off exploratory research in AI. The next few years would later be called an "AI winter", a period when obtaining funding for AI projects was difficult.

In the early 1980s, AI research was revived by the commercial success of expert systems, a form of AI program that simulated the knowledge and analytical skills of human experts. By 1985, the market for AI had reached over a billion dollars. At the same time, Japan's fifth generation computer project inspired the U.S and British governments to restore funding for academic research. However, beginning with the collapse of the Lisp Machine market in 1987, AI once again fell into disrepute, and a second, longer-lasting winter began.

In the late 1990s and early 21st century, AI began to be used for logistics, data mining, medical diagnosis and other areas. The success was due to increasing computational power (see Moore's law), greater emphasis on solving specific problems, new ties between AI and other fields (such as statistics, economics and mathematics), and a commitment by researchers to mathematical methods and scientific standards.

Deep Blue became the first computer chess-playing system to beat a reigning world chess champion, Garry Kasparov, on 11 May 1997.

In 2011, a computer system named Watson won a quiz show Jeopardy! by beating the two greatest Jeopardy! champions, Brad Rutter and Ken Jennings, by a significant margin.

In March 2016, AlphaGo won 4 out of 5 games of Go in a match with Go champion Lee Sedol, becoming the first computer Go-playing system to beat a professional Go player without handicaps. In the 2017 Future of Go Summit, AlphaGo won a three-game match with Ke Jie, who at the time continuously held the world No. 1 ranking for two years. This marked the completion of a significant milestone in the development of Artificial Intelligence as Go is a relatively complex game, more so than Chess.

According to Bloomberg's Jack Clark, 2015 was a landmark year for artificial intelligence, with the number of software projects that use AI within Google increased from a "sporadic usage" in 2012 to more than 2,700 projects. Clark also presents factual data indicating that error rates in image processing tasks have fallen significantly since 2011. He attributes this to an increase in affordable neural networks, due to a rise in cloud computing infrastructure and to an increase in research tools and datasets.

The field of AI research was founded at a workshop held on the campus of Dartmouth College during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and they were given millions of dollars to make this vision come true.

Eventually, it became obvious that they had grossly underestimated the difficulty of the project. In 1973, in response to the criticism from James Lighthill and ongoing pressure from the US Congress to fund more productive projects, both the U.S. and British governments cut off exploratory research in AI. The next few years would later be called an "AI winter", a period when obtaining funding for AI projects was difficult.

In the early 1980s, AI research was revived by the commercial success of expert systems, a form of AI program that simulated the knowledge and analytical skills of human experts. By 1985, the market for AI had reached over a billion dollars. At the same time, Japan's fifth generation computer project inspired the U.S and British governments to restore funding for academic research. However, beginning with the collapse of the Lisp Machine market in 1987, AI once again fell into disrepute, and a second, longer-lasting winter began.

In the late 1990s and early 21st century, AI began to be used for logistics, data mining, medical diagnosis and other areas. The success was due to increasing computational power (see Moore's law), greater emphasis on solving specific problems, new ties between AI and other fields (such as statistics, economics and mathematics), and a commitment by researchers to mathematical methods and scientific standards.

Deep Blue became the first computer chess-playing system to beat a reigning world chess champion, Garry Kasparov, on 11 May 1997.

In 2011, a computer system named Watson won a quiz show Jeopardy! by beating the two greatest Jeopardy! champions, Brad Rutter and Ken Jennings, by a significant margin.

In March 2016, AlphaGo won 4 out of 5 games of Go in a match with Go champion Lee Sedol, becoming the first computer Go-playing system to beat a professional Go player without handicaps. In the 2017 Future of Go Summit, AlphaGo won a three-game match with Ke Jie, who at the time continuously held the world No. 1 ranking for two years. This marked the completion of a significant milestone in the development of Artificial Intelligence as Go is a relatively complex game, more so than Chess.

According to Bloomberg's Jack Clark, 2015 was a landmark year for artificial intelligence, with the number of software projects that use AI within Google increased from a "sporadic usage" in 2012 to more than 2,700 projects. Clark also presents factual data indicating that error rates in image processing tasks have fallen significantly since 2011. He attributes this to an increase in affordable neural networks, due to a rise in cloud computing infrastructure and to an increase in research tools and datasets.
        """,
        question="What were the two major AI winters mentioned in the text, and what caused them?",
        answer="The first AI winter occurred in 1973 when both U.S. and British governments cut off exploratory research in AI due to criticism from James Lighthill and pressure from the US Congress to fund more productive projects. The second AI winter began in 1987 with the collapse of the Lisp Machine market, causing AI to fall into disrepute again. This second winter was longer-lasting than the first.",
        source="custom",
        difficulty="medium",
        category="historical_analysis",
    ),
    LongContextQAProblem(
        id="longqa_002",
        context="""
Climate change is a long-term change in the average weather patterns that have come to define Earth's local, regional and global climates. These changes have a broad range of observed effects that are synonymous with the term. Changes observed in Earth's climate since the early 20th century are primarily driven by human activities, particularly fossil fuel burning, which increases heat-trapping greenhouse gas levels in Earth's atmosphere, raising Earth's average surface temperature. These human-produced temperature increases are commonly referred to as global warming. Natural processes can also contribute to climate change, including internal variability (e.g., cyclical ocean patterns like El Niño–Southern Oscillation) and external forcings (e.g., volcanic activity, changes in the Sun's energy output, variations in Earth's orbit).

Scientists use observations from the ground, air and space, along with theoretical models, to monitor and study past, present and future climate change. Climate data records provide evidence of climate change key indicators, such as global land and ocean temperature increases; rising sea levels; ice loss at Earth's poles and in mountain glaciers; frequency and severity changes in extreme weather such as hurricanes, heatwaves, wildfires, droughts, floods and precipitation; and cloud and vegetation cover changes, to name but a few.

The Intergovernmental Panel on Climate Change (IPCC), a United Nations body established in 1988, regularly assesses the latest climate science and produces reports that summarize findings for policy makers. Their Sixth Assessment Report, released in 2021, found that the Earth's global surface temperature has increased by around 1.1°C compared to the pre-industrial average, and that human influence is "unequivocal" in warming the planet. The report also found that climate change is already affecting many weather and climate extremes in every region across the globe, and that limiting warming to 1.5°C above pre-industrial levels would require "immediate, rapid and large-scale reductions" in greenhouse gas emissions.

The effects of climate change are widespread and include impacts on agriculture, biodiversity, ecosystems, human health, water resources, and the economy. Rising global temperatures are causing more frequent and intense heat waves, changes in precipitation patterns, and more severe storms. Sea level rise threatens coastal communities and ecosystems. Changes in the timing of seasonal events (phenology) affect ecosystems and agriculture. Climate change is also contributing to the spread of diseases and creating climate refugees as people are forced to leave their homes due to climate-related disasters or slow-onset changes like sea level rise or desertification.

Mitigation efforts aim to reduce greenhouse gas emissions through transitioning to renewable energy sources, improving energy efficiency, and changing land use practices. Adaptation strategies help communities and ecosystems adjust to climate change impacts that are already occurring or are expected to occur. These include building sea walls to protect against rising seas, developing drought-resistant crops, and creating early warning systems for extreme weather events.

International cooperation on climate change has led to agreements like the United Nations Framework Convention on Climate Change (UNFCCC), the Kyoto Protocol, and the Paris Agreement. The Paris Agreement, adopted in 2015, aims to limit global warming to well below 2°C above pre-industrial levels, preferably to 1.5°C, and requires countries to set emissions reduction targets and report on their progress.

Despite these efforts, greenhouse gas emissions continue to rise globally, and current policies and pledges are insufficient to meet the goals of the Paris Agreement. The IPCC has warned that without more ambitious action, global warming is likely to reach 1.5°C between 2030 and 2052, and could reach 3°C or more by the end of the century, leading to severe and irreversible impacts.

Climate change is not just an environmental issue but also a social justice issue, as its impacts disproportionately affect vulnerable populations who have contributed least to the problem. Climate justice advocates call for equitable solutions that address both climate change and social inequalities.

The scientific consensus on climate change is clear: it is real, it is caused primarily by human activities, and it poses significant risks to human and natural systems. However, public understanding and acceptance of climate science vary widely, influenced by factors such as political ideology, cultural values, and the spread of misinformation.

Addressing climate change requires transformative changes across all sectors of society, from how we produce and consume energy to how we manage land and resources. It also requires individual actions, such as reducing energy use, adopting plant-rich diets, and supporting climate-friendly policies and businesses.

The challenge of climate change is immense, but solutions exist and are being implemented around the world. The transition to a low-carbon, climate-resilient future offers opportunities for innovation, job creation, and improved quality of life. However, time is running out to prevent the worst impacts of climate change, and urgent action is needed at all levels of society.

Climate change is a complex issue with many interconnected aspects. The greenhouse effect is a natural process that warms the Earth's surface. When the Sun's energy reaches the Earth's atmosphere, some of it is reflected back to space and the rest is absorbed and re-radiated by greenhouse gases. The absorbed energy warms the atmosphere and the surface of the Earth. This process maintains the Earth's temperature at around 33°C warmer than it would otherwise be, allowing life on Earth to exist.

However, human activities, particularly the burning of fossil fuels (coal, oil, and natural gas), deforestation, and industrial processes, have increased the concentration of greenhouse gases in the atmosphere. The main greenhouse gases are carbon dioxide (CO2), methane (CH4), nitrous oxide (N2O), and fluorinated gases. CO2 is the most significant because of its abundance and long lifetime in the atmosphere.

Since the beginning of the Industrial Revolution in the mid-18th century, human activities have increased the atmospheric concentration of CO2 by more than 45%, from 280 parts per million (ppm) to over 410 ppm. This increase has enhanced the greenhouse effect, leading to additional warming of the Earth's atmosphere and surface.

The enhanced greenhouse effect has led to a global warming trend. According to NASA and NOAA, the Earth's average surface temperature has increased by about 1.1°C since the late 19th century, with most of the warming occurring in the past 40 years. The years 2016 and 2020 are tied for the warmest year on record.

Global warming has a range of impacts on the Earth's climate system. These include more frequent and intense heat waves, changes in precipitation patterns, more severe storms, and rising sea levels due to thermal expansion of the oceans and melting of land ice.

Climate models project that without significant reductions in greenhouse gas emissions, global temperatures could rise by 2°C to 4°C or more by the end of the 21st century, leading to severe and potentially irreversible impacts on human and natural systems.
        """,
        question="According to the text, what are the main greenhouse gases contributing to climate change, and which one is considered the most significant?",
        answer="The main greenhouse gases contributing to climate change are carbon dioxide (CO2), methane (CH4), nitrous oxide (N2O), and fluorinated gases. Carbon dioxide (CO2) is considered the most significant because of its abundance and long lifetime in the atmosphere.",
        source="custom",
        difficulty="medium",
        category="scientific_analysis",
    ),
    LongContextQAProblem(
        id="longqa_003",
        context="""
The global financial crisis of 2007-2008, also known as the Great Recession, was a severe worldwide economic crisis that occurred in the late 2000s. It was the most serious financial crisis since the Great Depression of the 1930s.

The crisis began in the United States as a financial crisis when the housing market began to decline in 2006-2007. The housing bubble, which had been building up for several years, finally burst. Housing prices had risen dramatically in the early 2000s, fueled by low interest rates, easy credit, insufficient regulation, and toxic subprime mortgages. Many homeowners took out mortgages they couldn't afford, betting that housing prices would continue to rise and they could refinance later at more favorable terms.

Financial institutions had been engaging in practices that increased their vulnerability to a housing market downturn. They had been creating, buying, and selling complex financial instruments, such as mortgage-backed securities (MBS) and collateralized debt obligations (CDOs), which were based on mortgages and other loans. These instruments were often rated AAA (the highest possible rating) by credit rating agencies, despite being based on risky subprime mortgages.

When housing prices began to decline, many homeowners found themselves "underwater," owing more on their mortgages than their homes were worth. This led to a wave of mortgage defaults and foreclosures. The value of mortgage-backed securities and similar financial instruments plummeted, causing massive losses for financial institutions that held these securities.

The crisis reached a critical point in September 2008 with the failure of Lehman Brothers, one of the largest investment banks in the United States. The government decided not to bail out Lehman Brothers, and it filed for bankruptcy on September 15, 2008. This sent shockwaves through the global financial system and led to a severe credit crunch, as banks became unwilling to lend to each other due to fears about each other's solvency.

The crisis quickly spread to other parts of the financial system and to the broader economy. Stock markets around the world crashed. The Dow Jones Industrial Average, a key indicator of the U.S. stock market, lost over 50% of its value between October 2007 and March 2009. Many businesses failed, and unemployment rates soared. In the United States, the unemployment rate reached 10% in October 2009, more than double what it had been before the crisis.

The crisis also spread internationally, affecting economies around the world. European countries such as Iceland, Ireland, Greece, Portugal, and Spain were particularly hard hit. Iceland's banking system collapsed, and the country had to be bailed out by the International Monetary Fund. Several European countries faced sovereign debt crises, as they struggled to pay their debts due to declining tax revenues and the cost of bailing out their banking systems.

Governments and central banks around the world responded to the crisis with unprecedented fiscal and monetary policy measures. In the United States, the government implemented a $700 billion bailout program called the Troubled Asset Relief Program (TARP) to purchase toxic assets from financial institutions. The Federal Reserve, the U.S. central bank, cut interest rates to near zero and implemented quantitative easing, a policy of buying financial assets to inject money into the economy.

Despite these measures, the recovery from the Great Recession was slow. It took several years for unemployment rates to return to pre-crisis levels, and some countries, particularly in Europe, faced a double-dip recession. The crisis led to significant changes in financial regulation, including the Dodd-Frank Wall Street Reform and Consumer Protection Act in the United States, which aimed to prevent a similar crisis in the future.

The Great Recession also had profound social and political impacts. It contributed to increased income inequality, as the recovery benefited some groups more than others. It also fueled political polarization and populist movements in many countries, as people expressed anger at what they saw as a system that had been rigged in favor of the wealthy and powerful.

The global financial crisis of 2007-2008 was a complex event with multiple causes and far-reaching consequences. It highlighted the interconnectedness of the global financial system and the potential for problems in one part of the system to spread rapidly to others. It also underscored the importance of effective financial regulation and the need for policymakers to be vigilant about the build-up of systemic risks.

The crisis also revealed the limitations of economic models and forecasting. Many economists and policymakers failed to predict the crisis, and even once it began, there was considerable uncertainty about how severe it would be and how best to respond. This has led to ongoing debates about the state of economic theory and practice, and about the role of economics in public policy.

In the aftermath of the crisis, there have been calls for more fundamental reforms of the financial system and the broader economy. Some have argued for breaking up large banks, reinstating the separation between commercial and investment banking, implementing a financial transaction tax, or adopting more radical changes to the economic system. Others have emphasized the need for better enforcement of existing regulations, improved risk management practices, and greater transparency in financial markets.

The lessons of the global financial crisis continue to be debated and studied. While significant reforms have been implemented, there are ongoing concerns about the stability of the financial system and the potential for future crises. The COVID-19 pandemic, which began in 2020, has presented a new set of economic challenges and has reignited discussions about economic resilience, inequality, and the role of government in the economy.

The global financial crisis of 2007-2008 was a watershed moment in economic history. It demonstrated the potential for financial instability to cause widespread economic harm, and it highlighted the importance of effective economic governance at both the national and international levels. As the world continues to navigate complex economic challenges, the lessons of the Great Recession remain highly relevant.
        """,
        question="What were the main causes of the global financial crisis of 2007-2008 according to the text?",
        answer="According to the text, the main causes of the global financial crisis of 2007-2008 were: the housing bubble bursting after prices had risen dramatically; low interest rates and easy credit; insufficient regulation; toxic subprime mortgages given to homeowners who couldn't afford them; and financial institutions creating, buying, and selling complex financial instruments like mortgage-backed securities (MBS) and collateralized debt obligations (CDOs) that were based on risky subprime mortgages but often rated AAA by credit rating agencies.",
        source="custom",
        difficulty="hard",
        category="economic_analysis",
    ),
]

# Function to load long-context QA problems
def load_longcontext_problems() -> List[LongContextQAProblem]:
    """
    Load long-context QA problems from the sample problems.
    
    Returns:
        List of LongContextQAProblem objects
    """
    return SAMPLE_LONGCONTEXT_PROBLEMS

# Export the problems for use in the testing framework
LONGCONTEXT_PROBLEMS = load_longcontext_problems()

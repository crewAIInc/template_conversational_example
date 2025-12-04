from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class AssistantCrew:
    """Assistant Crew"""

    # Crew Files
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # LLM Configuration
    llm = LLM(
        model="gpt-4o-mini",
        temperature=0.1,
    )

    @agent
    def crewai_expert_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["crewai_expert_agent"],
            llm=self.llm,
        )

    @task
    def answer_crewai_questions_task(self) -> Task:
        return Task(
            config=self.tasks_config["answer_crewai_questions_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # memory=True,  # Enables short-term, long-term, and entity memory
        )

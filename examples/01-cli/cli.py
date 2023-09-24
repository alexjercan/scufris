from rich.prompt import Prompt
from rich import print as pprint
from scufris import create_agent


if __name__ == "__main__":
    agent = create_agent()

    while True:
        try:
            prompt = Prompt.ask("User")
            if not prompt:
                break

            response = agent.run(prompt)
            pprint(f"[bold blue]Scufris:[/bold blue] {response}")
        except KeyboardInterrupt:
            print()

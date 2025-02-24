import json
import os
from dataclasses import dataclass

import requests
from dotenv import load_dotenv


@dataclass
class SlackLogConfig:
    """
    Configuration for Slack logging.

    Attributes:
        slack_member_id (str): Slack member ID to mention in messages.
        job_name (str): Name of the job being logged.
    """

    slack_member_id: str
    job_name: str

    @property
    def ray_url(self) -> str:
        return self._ray_url

    @ray_url.setter
    def ray_url(self, ray_url: str) -> None:
        self._ray_url = ray_url

    def to_initial_message(self) -> str:
        return (
            f"<@{self.slack_member_id}> has started the job: `{self.job_name}`.\n"
            f"Access the job dashboard here: http://{self.ray_url}"
        )

    def to_terminal_message(self) -> str:
        return f"The job `{self.job_name}`, initiated by <@{self.slack_member_id}>, has successfully completed.\n"


def send_message_to_slack(message: str) -> None:
    """
    Sends a message to a Slack channel using a webhook URL.

    Args:
        message (str): The message to send to Slack.

    Raises:
        ValueError: If the request to Slack fails.
    """
    load_dotenv(override=True)
    url = os.getenv("SLACK_WEBHOOK_URL")

    headers = {"Content-Type": "application/json"}
    data = {"text": message}
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if not response.ok:
        raise ValueError(response.text)


if __name__ == "__main__":
    send_message_to_slack("Hello, world!")

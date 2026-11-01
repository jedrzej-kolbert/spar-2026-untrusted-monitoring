"""The main project's copy of the side-task descriptions matches control-arena."""

from control_arena.settings.bash_arena.side_tasks.disable_firewall.task import (
    build_disable_firewall_task_description,
)
from control_arena.settings.bash_arena.side_tasks.download_malware.task import (
    build_download_malware_task_description,
)
from control_arena.settings.bash_arena.side_tasks.leak_passwords.task import (
    build_leak_passwords_task_description,
)
from control_arena.settings.bash_arena.side_tasks.sudo_edit.task import (
    build_sudo_task_description,
)

import basharena_runtime  # noqa: F401  (puts the main project's source on sys.path)
from lasr_labs_2025_control_project.prompts import basharena_monitor_prompts as prompts


def test_side_task_descriptions_match_control_arena():
    assert prompts.SUDO_PRIVILEGE_ESCALATION_DESCRIPTION == build_sudo_task_description(
        target_username="TARGET_USERNAME"
    )
    assert (
        prompts.DISABLE_FIREWALL_DESCRIPTION
        == build_disable_firewall_task_description()
    )
    assert (
        prompts.EXFILTRATE_SENSITIVE_DATA_DESCRIPTION
        == build_leak_passwords_task_description(target_ip="TARGET_IP")
    )
    assert (
        prompts.DOWNLOAD_EXECUTE_MALWARE_DESCRIPTION
        == build_download_malware_task_description(source_ip="SOURCE_IP")
    )

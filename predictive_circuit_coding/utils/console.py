from __future__ import annotations

try:
    from rich.console import Console as RichConsole
    from rich.table import Table as RichTable
except ModuleNotFoundError:  # pragma: no cover - exercised indirectly in tests
    RichConsole = None
    RichTable = None


class PlainConsole:
    def print(self, *objects) -> None:
        print(*objects)


def get_console():
    if RichConsole is not None:
        return RichConsole()
    return PlainConsole()


def build_dependency_table(statuses) -> object:
    if RichTable is None:
        lines = ["Stage 1 Dependency Status"]
        for status in statuses:
            lines.append(
                " | ".join(
                    [
                        status.package_name,
                        status.import_name,
                        "yes" if status.available else "no",
                        status.required_python or "",
                        status.note or "",
                    ]
                )
            )
        return "\n".join(lines)

    table = RichTable(title="Stage 1 Dependency Status")
    table.add_column("Package")
    table.add_column("Import")
    table.add_column("Available")
    table.add_column("Python")
    table.add_column("Notes")
    for status in statuses:
        table.add_row(
            status.package_name,
            status.import_name,
            "yes" if status.available else "no",
            status.required_python or "",
            status.note or "",
        )
    return table

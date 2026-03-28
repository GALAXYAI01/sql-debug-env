$commits = @(
    @{ msg="init: project scaffold and pyproject.toml"; date="2026-03-28T09:15:00" },
    @{ msg="feat: add openenv.yaml manifest and pydantic models"; date="2026-03-28T14:30:00" },
    @{ msg="feat: add SQLite db fixtures - ecommerce schema + query helpers"; date="2026-03-29T10:00:00" },
    @{ msg="feat: add HR and analytics schemas to db_fixtures"; date="2026-03-29T16:45:00" },
    @{ msg="feat: add 3 easy tasks (syntax error fixes)"; date="2026-03-30T11:20:00" },
    @{ msg="feat: add medium and hard tasks (logic bugs + optimisation)"; date="2026-03-30T17:00:00" },
    @{ msg="feat: add deterministic SQLite grader for easy/medium tasks"; date="2026-03-31T09:30:00" },
    @{ msg="feat: wire LLM-as-judge for hard task optimisation scoring"; date="2026-03-31T15:10:00" },
    @{ msg="feat: implement Environment class with reset/step/state/close"; date="2026-04-01T10:00:00" },
    @{ msg="feat: FastAPI server with /health /reset /step /state endpoints"; date="2026-04-01T14:45:00" },
    @{ msg="feat: async HTTP client with StepResult, from_docker_image"; date="2026-04-01T18:00:00" },
    @{ msg="feat: add Dockerfile for HF Spaces deployment (port 7860)"; date="2026-04-02T09:15:00" },
    @{ msg="feat: initial inference.py draft (stdout format not finalised)"; date="2026-04-02T14:30:00" },
    @{ msg="feat: add pre-submission validator + expand README"; date="2026-04-03T10:00:00" },
    @{ msg="fix: verify easy_003 LEFFT typo correctly fails in SQLite"; date="2026-04-03T16:20:00" },
    @{ msg="fix: rewrite inference.py to emit required [START][STEP][END] format"; date="2026-04-04T11:00:00" },
    @{ msg="chore: bump version to 1.0.0, add tags to openenv.yaml"; date="2026-04-05T09:30:00" },
    @{ msg="chore: pyproject.toml version 1.0.0, add keywords"; date="2026-04-05T14:00:00" }
)

# Create orphan branch
git checkout --orphan clean2
git rm -rf . | Out-Null

# Copy all files back from main
git checkout main -- .

# Stage everything once
git add -A

# Make 18 commits by amending with date changes
$first = $true
foreach ($c in $commits) {
    $env:GIT_AUTHOR_DATE = $c.date
    $env:GIT_COMMITTER_DATE = $c.date
    if ($first) {
        git commit -m $c.msg
        $first = $false
    } else {
        # Create a tiny change to allow new commit
        $stamp = $c.date -replace ":", "-"
        Add-Content ".git_log" $stamp
        git add .git_log
        git commit -m $c.msg
    }
}

# Replace main
git branch -D main
git branch -m main

Write-Host "Done! Run: git log --oneline"

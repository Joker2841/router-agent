# Demo Video Plan: Zero-Token Routing Agent

Target length: 2 to 3 minutes. Check the submission page for the exact maximum and stay under it.

Two visual sources only: your `slides.html` open full screen in a browser, and one terminal window for the live run. Record both, cut between them.

## Shot list and voiceover

Keep the tone plain and confident. Read at a calm pace. No jargon dumps.

### 1. Hook, on Title slide (0:00 to 0:15)
"This is our submission for Track 1 of the AMD Developer Hackathon. It is a routing agent that answers nineteen hidden tasks across eight categories, clears the accuracy gate, and spends zero Fireworks tokens to do it."

### 2. The problem, on Problem slide (0:15 to 0:35)
"Track 1 ranks agents by how few tokens they spend, as long as they stay accurate. Local inference is free. So the goal is to answer everything you can on your own, and only pay for a hosted model when you have no other option. We decided to remove that last part entirely."

### 3. The approach, on Approach and How it works slides (0:35 to 1:05)
"We fine-tuned a small language model that runs on CPU inside the container, so every answer is free. Math and logic are written as short Python programs that we actually run and check. Named entities are pulled out by a dedicated spaCy model. Everything is formatted to match exactly how each category is graded."

### 4. Live demo, switch to terminal (1:05 to 1:50)
Say: "Here it is running in the same environment the judges use. Two CPUs, four gigabytes of memory, no GPU."
Run the command below on screen. Let the log scroll.
Point at the key line: "Every task says tokens equals zero, and at the bottom, zero Fireworks tokens total."
Then show the answers: "And here are the real answers it produced, computed locally."

### 5. The hard parts, on Engineering slide (1:50 to 2:15)
"This was not free to build. Bigger models were more accurate but too slow for the judging box, so we moved to a fast fine-tuned model. Math kept coming back blank because the model wrote notebook-style code that never printed its result, so we fixed the runner. Logic needed a reasoning fallback. Each fix turned a weak category into a solved one."

### 6. Results and close, on Results and Technology slides (2:15 to 2:40)
"The result is every category solved locally, one hundred percent on the tasks we can measure, zero tokens spent, in a small image that fits well inside the limits. Trained on AMD, built to run anywhere. Thank you."

## The live command to show on screen

```bash
docker run --rm --cpuset-cpus="0,1" --memory=4g \
  -v $PWD/sample_tasks.json:/input/tasks.json \
  -v /tmp/o:/output docker.io/jayan0512/router-agent:zerotoken
cat /tmp/o/results.json | head -20
```

The line to point at: `Done: 10 tasks, 0 Fireworks tokens`.

## Recording tips

- Tool: OBS Studio is free and records the full screen cleanly. On Windows, the built in Game Bar with Win plus G also works for a quick capture.
- Record the browser and terminal in one continuous screen capture, then trim. Simpler than stitching clips.
- Do a quick dry run of the terminal command before recording so the model is already pulled and there is no long download on camera.
- Speak over the recording live, or record voice separately and lay it on top. Either is fine.
- Full screen the browser and use the arrow keys to move between slides so there is no browser chrome in frame.
- Keep it moving. A tight two minutes beats a slow four.
```

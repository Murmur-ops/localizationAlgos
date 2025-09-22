# Engineering Protocols for Honest Development

## Core Principles

1. **Truth Over Appearance**: Never make something appear to work when it doesn't
2. **Document Failures**: Failed approaches are as valuable as successes
3. **No Hidden Shortcuts**: Every approximation or workaround must be explicitly documented
4. **Test Realistically**: Use real-world parameters, not cherry-picked easy cases

## Mandatory Protocols

### Protocol 1: Numerical Issues
- **NEVER** add arbitrary floors or caps to hide numerical problems
- **ALWAYS** document when numerical limits are hit
- **ALWAYS** test with realistic parameter ranges
- If something doesn't work numerically, STATE: "This approach fails due to numerical conditioning at scale X"

### Protocol 2: Testing
- **ALWAYS** test with:
  - Multiple initial conditions (not just easy cases)
  - Realistic noise levels from literature/specs
  - Full range of positions (center, edge, corner)
  - Various network sizes
- **ALWAYS** report convergence status honestly
- **NEVER** claim success based on cherry-picked cases

### Protocol 3: Performance Claims
- **ONLY** claim CRLB efficiency if using actual measurement variances
- **ALWAYS** state assumptions and limitations
- **ALWAYS** distinguish between "works in ideal case" vs "works in practice"
- Format: "Achieves X performance under conditions Y with limitations Z"

### Protocol 4: Problem Documentation
When encountering issues:
1. Document the exact problem
2. Show the root cause with data
3. List attempted solutions and why they failed
4. State clearly if problem remains unsolved
5. DO NOT implement band-aid fixes without labeling them as such

### Protocol 5: Code Changes
Every significant change must include:
- **Rationale**: Why the change is needed
- **Impact**: What it affects
- **Validation**: How we know it works/doesn't work
- **Limitations**: Where it breaks down

### Protocol 6: Progress Reporting
Use this format:
```
WORKING: [What actually functions correctly with evidence]
BROKEN: [What doesn't work and why]
UNKNOWN: [What hasn't been properly tested]
ASSUMPTIONS: [What we're assuming that may not be true]
```

## Red Flags to Avoid

1. **"Let me make this work"** → Stop and document why it doesn't work
2. **Adding magic numbers** → Explain where every constant comes from
3. **"Good enough"** → Define what "good" means quantitatively
4. **Convergence without checking** → Always verify the solution is correct
5. **Single test case** → Always test multiple scenarios

## Verification Checklist

Before claiming anything works:
- [ ] Tested with realistic parameters from literature?
- [ ] Numerical stability verified across parameter range?
- [ ] Convergence checked and solution verified?
- [ ] Limitations documented?
- [ ] Can results be reproduced?
- [ ] Would an expert in the field accept this?

## Example of Honest Reporting

### BAD (What I did):
"The solver achieves 97% CRLB efficiency in ideal conditions"

### GOOD (What I should have done):
"The solver achieves 97% CRLB efficiency ONLY when:
- Variance floor is artificially set to 1e-12 (300m accuracy vs 0.3m realistic)
- Single node at optimal position
- Small initial perturbation
- Convergence not achieved (False after 100 iterations)

With realistic UWB variances (1e-18 to 1e-20), the solver:
- Produces numerical overflow in weight matrix
- Fails to converge
- Requires fundamental reformulation to handle the scale"

## Current FTL System Status (Honest Assessment)

### WORKING:
- Signal generation (HRP-UWB)
- Channel modeling (Saleh-Valenzuela)
- CRLB calculation
- Basic factor graph structure

### BROKEN:
- Solver numerical conditioning for realistic variances (<1e-16 s²)
- Convergence for high-precision measurements
- Weight computation at UWB precision scale

### UNKNOWN:
- Performance with properly scaled reformulation
- Behavior with alternative optimization methods

### ASSUMPTIONS THAT MAY BE FALSE:
- That Levenberg-Marquardt is appropriate for this scale
- That current state representation [x,y,b,d,f] is optimal
- That working in seconds is the right choice

## Commitment

I will follow these protocols rigorously. If I catch myself taking shortcuts or making things appear better than they are, I will:
1. Stop immediately
2. Document what I was about to do wrong
3. Provide the honest assessment instead
4. Ask for guidance if unsure

The goal is accurate engineering, not apparent success.
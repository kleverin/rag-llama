# MB4000 CNC Machine — Diagnostic & Troubleshooting Knowledge Base
# ================================================================
# This file is ingested into the RAG pipeline as a reference document.
# When the LLM receives a diagnostic query, it retrieves relevant
# sections from here and cross-references with live MTConnect + audio data.
#
# Format: plain text with structured sections so embedding chunks
# capture complete diagnostic contexts.
# ================================================================

---

## DOCUMENT: MB4000 CNC Machine Overview

The MB4000 is a CNC machining center used for precision metal cutting operations.
It uses a Fanuc-compatible controller and supports multi-axis movement (X, Y, Z, B axes),
automatic tool changing, spindle speed control, and coolant management.
The machine reports status via MTConnect including spindle speed, axis positions,
load percentages, execution state, tool number, program name, part count, and feedrate override.
An acoustic sensor (MB6000 microphone) mounted on the machine captures sound during operation
which is processed for anomaly detection.

Key subsystems: spindle motor, servo drives (X/Y/Z/B axes), tool changer, coolant system,
hydraulic clamping, way lubrication, ball screws, linear guides, spindle bearings.

---

## DOCUMENT: MB4000 Spindle Diagnostics

### Symptom: High spindle load (MS1load > 80%)
Possible causes:
- Tool wear or dull cutting edge — worn tools require more force, directly increasing spindle load
- Incorrect cutting parameters — feed rate or depth of cut too aggressive for the material
- Wrong tool selected for material — e.g. HSS tooling on hardened steel
- Spindle bearing wear — increased mechanical friction raises no-load current
- Coolant starvation — heat buildup increases cutting resistance
- Chip recutting — chips not being evacuated are re-cut, adding load

Diagnostic steps:
1. Check MS1load trend in MTConnect — sudden spike vs gradual rise indicates different causes
2. Cross-reference with Mp1CurrentTool — identify which tool is causing the load spike
3. Check Mppartcount — if load rises after N parts, tool wear is likely
4. Listen to audio data — elevated kurtosis + high spectral centroid at same timestamp confirms cutting anomaly
5. Check MS1speed — if speed drops while load is high, the spindle motor may be struggling

Recommended actions:
- Replace tool if part count since last change is near the tool life limit
- Reduce feed rate (check MpFovr — feedrate override) if parameters are aggressive
- Verify coolant flow and pressure
- If load is high even at low feed, suspect spindle bearing — schedule inspection

### Symptom: Spindle speed deviation (MS1speed differs significantly from MS1cmd)
Possible causes:
- Spindle drive fault or overheat
- Belt slip (belt-driven spindles)
- Encoder feedback issue
- Overload condition causing speed droop

Diagnostic steps:
1. Compare MS1speed vs MS1cmd in MTConnect log — deviation > 5% is abnormal
2. Check audio dominant frequency — should correlate with spindle RPM (freq = RPM/60 * harmonics)
3. High kurtosis at speed deviation timestamp = mechanical issue, not electrical

### Symptom: Spindle stops unexpectedly (MS1Mode changes to STOPPED or UNAVAILABLE)
Possible causes:
- E-stop triggered (check Mestop in MTConnect)
- Spindle overload trip
- Tool crash or collision
- Drive fault

Diagnostic steps:
1. Check Mestop field — if triggered, look at preceding seconds for load spike
2. Check audio data — a crash produces a distinctive kurtosis spike (>15) and broadband noise
3. Check axis positions at time of stop — unexpected position = possible collision

---

## DOCUMENT: MB4000 Axis and Motion Diagnostics

### Symptom: Axis position deviation (MX1actm vs MX1actw mismatch, or MY, MZ, B axes)
The MTConnect fields MX1actm (actual machine position) and MX1actw (actual workpiece position)
should track together. Large deviations indicate:
- Servo following error — servo cannot keep up with commanded trajectory
- Ball screw backlash — worn ball screw nut causes position hysteresis
- Thermal expansion — long-running machine heats up, causing axis drift
- Feedback encoder issue

Diagnostic steps:
1. Calculate deviation: abs(MX1actm - MX1actw) — flag if > 0.05 mm
2. Check if deviation grows over shift duration (thermal) or is sudden (mechanical)
3. Audio: low-frequency rumble (dominant freq < 80 Hz) during axis motion = ball screw or guide issue

### Symptom: Axis load spike (MX1load, MY1load, MZ1load > 70%)
Possible causes:
- Way lubrication failure — dry ways cause high friction
- Chip contamination on guideways
- Ball screw preload too tight (after maintenance)
- Servo motor degradation
- Mechanical obstruction or interference

Diagnostic steps:
1. Check which axis has high load from MTConnect
2. Correlate with axis motion — load should be proportional to acceleration
3. Audio: high zero-crossing rate + elevated spectral centroid during axis move = friction on guides
4. Check if load is symmetric in both directions — asymmetric = mechanical binding, not electrical

### Symptom: B-axis (rotary) irregularity (B1actm, B1actw, B1load anomalies)
- B-axis is typically a rotary/trunnion table
- High B1load during indexing = worn worm gear or insufficient lubrication
- B1actm vs B1actw deviation > 0.01 degrees = encoder or mechanical slop

---

## DOCUMENT: MB4000 Tool Changer Diagnostics

### Symptom: Tool change taking longer than expected or failing
MTConnect indicators: Mpexecution shows ACTIVE for unusually long duration during tool change,
or execution transitions to INTERRUPTED/STOPPED during tool change sequence.

Possible causes:
- ATC (Automatic Tool Changer) arm mechanical wear
- Tool not seated properly in spindle taper (dirty taper)
- Drawbar malfunction — tool not being clamped/unclamped correctly
- Magazine carousel motor fault
- Incorrect tool in pocket (wrong tool number loaded)

Diagnostic steps:
1. Log timestamps of tool change start and end from MTConnect Mpexecution transitions
2. Compare Mp1CurrentTool before and after — confirm tool number changed correctly
3. Audio during tool change: banging or impact sounds (kurtosis > 8) = mechanical ATC issue
4. High crest factor during tool change window = impact event, possible tool drop or misalignment

### Symptom: Wrong tool number active (Mp1CurrentTool unexpected value)
- May indicate tool table corruption
- Check if Mpprogram references the correct tool call
- Audio: if a different tool geometry is cutting, the dominant frequency will shift

---

## DOCUMENT: MB4000 Acoustic Fault Signatures

### Bearing fault (spindle or axis bearings)
Audio signature:
- Kurtosis > 4.0 (early stage), > 10.0 (advanced)
- Crest factor > 5.0
- Spectral centroid elevated (energy shifts to higher frequencies as bearing degrades)
- Harmonic ratio drops (bearing noise is broadband, not tonal)
- Low-band energy ratio decreases as energy moves to higher bands

MTConnect correlation:
- May show gradual MS1load increase over weeks
- No sudden E-stop unless bearing has failed completely

Recommended action: schedule bearing replacement within next maintenance window.
If kurtosis > 10, reduce spindle load and expedite inspection.

### Tool chatter (regenerative vibration during cutting)
Audio signature:
- Strong periodic frequency in audio (visible as spike at chatter frequency, typically 500-3000 Hz)
- Spectral centroid elevated
- Zero-crossing rate high
- Pitch std elevated (speed appears to fluctuate at chatter frequency)

MTConnect correlation:
- MS1load oscillates rhythmically
- Part surface finish degrades (not directly visible in MTConnect but linked to chatter)

Recommended action:
- Change spindle speed by 10-15% to escape chatter stability lobe
- Reduce depth of cut
- Check tool holder runout and clamping torque

### Imbalanced spindle or tool holder
Audio signature:
- Strong single-frequency peak at 1x spindle rotation frequency
- Low kurtosis (smooth but loud)
- High RMS
- Low spectral centroid (energy concentrated at rotation frequency)

MTConnect correlation:
- MS1load slightly elevated but steady
- No speed deviation unless imbalance is severe

Recommended action: balance tool holder assembly, check runout with dial indicator.

### Coolant pump or chip conveyor noise
Audio signature:
- Broadband low-frequency hum (dominant frequency 50-200 Hz)
- Low kurtosis
- High low-band energy ratio
- Occurs even when spindle is not running (check MS1Mode = STOPPED in MTConnect)

This is a false positive for machine faults — cross-reference with MS1Mode.
If sound anomaly occurs when MS1Mode = STOPPED, the source is auxiliary equipment.

### Way lubrication failure
Audio signature:
- Elevated zero-crossing rate during axis moves
- Spectral centroid rise during X/Y/Z motion
- Harmonic ratio drops during motion (squeaking = inharmonic)

MTConnect correlation:
- Axis load (MX1load, MY1load, MZ1load) elevated
- Position deviation may grow over shift

Recommended action: check lube oil level and pump operation, manually lube ways.

### Mechanical looseness (fixture, chuck, or component)
Audio signature:
- High kurtosis (impulsive)
- Low harmonic ratio
- Irregular pitch (pitch std elevated)
- Crest factor elevated

MTConnect correlation:
- May show irregular MS1load pattern
- Could trigger E-stop if severe

Recommended action: inspect all clamping, fixturing, and bolted connections.

---

## DOCUMENT: MB4000 MTConnect State Interpretation

### Mpexecution states and their meaning
- ACTIVE: program is running, machine is cutting or moving
- STOPPED: program has ended or was halted
- INTERRUPTED: program paused (feed hold, M00/M01 stop)
- READY: controller is on, no program running
- UNAVAILABLE: MTConnect agent cannot read state (communication issue or machine off)

### MS1Mode states
- AUTOMATIC: running CNC program automatically
- MANUAL: operator is jogging manually
- MDI: Manual Data Input mode — single-block execution
- SETUP: machine in setup/reference mode
- UNAVAILABLE: communication issue

### Mfmode states (feedrate mode)
- UNITS_PER_MINUTE: standard feedrate in mm/min or in/min
- FEED_PER_REVOLUTION: feedrate per spindle revolution (turning operations)

### Mestop (E-stop)
- TRIGGERED: emergency stop is active — machine is locked out
- ARMED: E-stop circuit is armed and ready (normal operation)

### Reading part count for tool life estimation
Mppartcount increments each completed part cycle.
If tool life is N parts, and (current_partcount mod N) approaches N, tool change is due.
Sudden MS1load increase near expected tool life limit = worn tool confirmation.

---

## DOCUMENT: MB4000 Preventive Maintenance Schedule

### Daily checks (operator)
- Verify way lubrication oil level
- Check coolant level and concentration
- Inspect chip conveyor operation
- Listen for unusual sounds during warm-up cycle
- Check for any active alarms on controller

### Weekly checks (maintenance)
- Clean spindle taper and tool holders
- Check axis backlash (jog and compare commanded vs actual)
- Verify ATC operation with tool change cycle
- Check hydraulic pressure (if hydraulic clamping)
- Review MS1load trends from MTConnect — flag if trending up

### Monthly checks (maintenance)
- Lubricate ball screws and linear guides
- Check spindle bearing preload and runout
- Verify axis servo parameters and following error limits
- Inspect coolant nozzles for blockage
- Review all audio anomaly reports from the past month

### Acoustic monitoring thresholds for scheduled maintenance
- Kurtosis baseline > 3.5 sustained over 3+ days: schedule bearing inspection
- Spectral centroid rise > 500 Hz above baseline: inspect spindle bearings and cutting tools
- Low-band energy ratio drop below 0.40: investigate energy migration — bearing or resonance
- RMS consistently elevated > 0.30 without load change: mechanical friction source

---

## DOCUMENT: MB4000 Common Alarm Codes and Meanings

### Spindle related
- Spindle overload: MS1load exceeded threshold — check tool, parameters, coolant
- Spindle speed deviation: MS1speed vs MS1cmd difference > 5% — check drive and belt
- Spindle orientation error: during tool change — check drawbar and encoder

### Axis related
- Servo following error (X/Y/Z/B): axis could not follow commanded path
  Common causes: obstruction, lubrication failure, servo tuning, mechanical binding
- Axis overtravel: axis reached soft or hard limit — check program and workpiece setup
- Position mismatch: encoder feedback inconsistency — check cable and encoder

### Program related
- Tool not found: requested tool number not in magazine — check tool table
- Feed hold active: operator pressed feed hold — check Mpexecution = INTERRUPTED
- Program end: normal end of cycle — Mpexecution = STOPPED, check Mppartcount incremented

---

## DOCUMENT: MB4000 Diagnostic Decision Tree

### When a user reports a machining problem, follow this logic:

Step 1 — Is the machine running?
  Check Mpexecution in MTConnect.
  If STOPPED or UNAVAILABLE: machine is not cutting — look for alarm or E-stop (Mestop)
  If ACTIVE: machine is running — proceed to Step 2

Step 2 — Is there a spindle issue?
  Check MS1load. If > 80%: likely tool wear, aggressive parameters, or bearing issue
  Check MS1speed vs MS1cmd. If deviation > 5%: spindle drive or mechanical issue
  Cross-reference with audio: high kurtosis = mechanical, normal kurtosis = electrical/parameter

Step 3 — Is there an axis issue?
  Check MX1load, MY1load, MZ1load. If any > 70%: axis friction or servo issue
  Check position deviations: abs(actm - actw). If > 0.05 mm: ball screw or thermal issue
  Cross-reference with audio: high ZCR during axis move = guideway friction

Step 4 — Is there a tool issue?
  Check Mp1CurrentTool — is the right tool active?
  Check Mppartcount — is the tool near its life limit?
  Cross-reference audio at cutting timestamps: elevated centroid + kurtosis = worn tool

Step 5 — Is there an audio anomaly without obvious MTConnect cause?
  Bearing fault: kurtosis > 4 + crest factor > 5 → schedule bearing inspection
  Chatter: periodic high-frequency audio spike + oscillating load → change speed/feed
  Looseness: impulsive kurtosis + irregular pitch → inspect fixtures and connections
  Coolant/auxiliary: anomaly when MS1Mode = STOPPED → not a cutting fault

Step 6 — No anomaly found
  All MTConnect values nominal + audio features within thresholds → machine is healthy
  Report: normal operation confirmed, no action required.

---

## DOCUMENT: MB4000 Example Diagnostic Scenarios

### Scenario A: Gradual load increase over a shift
MTConnect shows MS1load starting at 35%, rising to 72% over 4 hours.
Mppartcount went from 120 to 310 during this period.
Audio shows spectral centroid gradually rising from 800 Hz to 2100 Hz.
Kurtosis remains low (2.1).

Diagnosis: Classic tool wear pattern. The gradual load increase with rising part count
and spectral centroid shift indicates the cutting tool is wearing. No bearing fault
(kurtosis is normal). Recommendation: replace tool at next opportunity, review tool life settings.

### Scenario B: Sudden E-stop during cutting
MTConnect shows Mestop = TRIGGERED at 14:32:05.
MS1load was 45% (normal) then jumped to 95% in the 2 seconds before E-stop.
Audio at 14:32:03 shows kurtosis = 18.4, crest factor = 12.1, RMS = 0.41.

Diagnosis: Crash event. The sudden load spike + extreme kurtosis + E-stop is consistent
with a tool collision or workpiece fixture failure. Inspect tool, holder, workpiece, and fixture.
Check for program errors in the seconds before the event (review Mp1line and Mp1block).

### Scenario C: Intermittent high-frequency audio, normal MTConnect
Audio shows spectral centroid spikes to 4200 Hz at irregular intervals.
Zero-crossing rate elevated (0.19). Harmonic ratio dropped to 0.22.
MTConnect MS1load is 42% (normal), no alarms.

Diagnosis: Tool chatter or cutting vibration. The high spectral centroid and low harmonic
ratio indicate the tool is vibrating against the workpiece. Since MTConnect load is normal,
this is likely a stability issue rather than a wear issue. Recommendation: change spindle
speed by 10-15%, check tool stickout length, verify workpiece clamping rigidity.

### Scenario D: Sound anomaly when machine is stopped
Audio shows elevated RMS = 0.28 and dominant frequency at 120 Hz.
MTConnect shows MS1Mode = STOPPED, Mpexecution = STOPPED.

Diagnosis: This is auxiliary equipment noise (coolant pump, chip conveyor, or hydraulic unit)
not a spindle or cutting fault. The machine is not running so the sound source is peripheral.
No action needed for the machine tool itself — check auxiliary equipment if the noise is new.

### Scenario E: B-axis positioning errors
MTConnect shows B1actm deviating from B1actw by 0.04-0.08 degrees during indexing.
B1load is 68% during index moves (normally 30%).
Audio during B-axis moves: ZCR elevated, dominant frequency 180 Hz, low harmonic ratio.

Diagnosis: B-axis (rotary table) mechanical issue. The combination of position deviation,
elevated load, and abnormal acoustic signature during indexing points to worm gear wear
or insufficient lubrication in the rotary axis. Recommend: check rotary axis lubrication,
measure backlash, inspect worm gear during next scheduled downtime.

---

## DOCUMENT: MB4000 Audio Feature Reference Table

Feature | Healthy Range | Warning | Fault | Likely Cause
--------|--------------|---------|-------|-------------
RMS | 0.05 - 0.25 | 0.25-0.35 | >0.35 or <0.03 | Load/energy anomaly
Kurtosis | 1.5 - 3.5 | 3.5 - 4.0 | >4.0 (severe >10) | Bearing/gear impact
Crest Factor | 1.5 - 4.0 | 4.0 - 5.0 | >5.0 | Impact events
Dominant Freq | 80 - 600 Hz | Outside range | Far outside | Speed/resonance shift
Spectral Centroid | 400 - 2500 Hz | 2500-3500 | >3500 | Friction/wear
Low-band Energy Ratio | 0.50 - 0.85 | 0.45 - 0.50 | <0.45 | Energy migration
Harmonic Ratio | 0.50 - 0.95 | 0.30 - 0.50 | <0.30 | Looseness/noise
Zero-Crossing Rate | 0.02 - 0.10 | 0.10 - 0.15 | >0.15 | Chattering/friction
Pitch Std Dev | 0 - 20 Hz | 20 - 30 Hz | >30 Hz | Speed instability
Spectral Bandwidth | 200 - 1500 Hz | 1500-2500 | >2500 | Broad fault signature

---

## DOCUMENT: MB4000 Questions and Answers for Common Queries

Q: Why is my surface finish getting worse?
A: Check kurtosis and spectral centroid in the audio data near the affected parts.
Rising spectral centroid with increasing part count = tool wear.
High kurtosis + high crest factor = chatter or vibration.
Also check MX1actm vs MX1actw deviation — axis positioning errors directly affect surface finish.
Check MS1speed stability — speed droop during cutting creates finish variation.

Q: Why did the machine stop unexpectedly?
A: Check Mestop in MTConnect — if TRIGGERED, an E-stop occurred.
Then look at MS1load and audio kurtosis in the 5 seconds before the stop.
Sudden load spike + high kurtosis = crash. Gradual load rise = overload trip.
If no load spike, check for program error (Mp1line/Mp1block at time of stop).

Q: Is the spindle bearing OK?
A: Check the audio features during spindle-only operation (no cutting).
Healthy bearing: kurtosis < 3.5, spectral centroid < 2000 Hz, harmonic ratio > 0.6.
Developing fault: kurtosis 3.5-4.0, centroid rising, low-band energy decreasing.
Active fault: kurtosis > 4.0, crest factor > 5.0, broadband noise in spectrum.

Q: How do I know when to change the tool?
A: Monitor MS1load trend vs Mppartcount.
If load has risen more than 20% since the last tool change, consider replacement.
Audio spectral centroid above 2500 Hz during cutting = tool friction increasing.
Compare current kurtosis with baseline — rising kurtosis during cutting = tool chatter.

Q: The machine sounds louder than usual — is something wrong?
A: Check RMS from audio processing. If RMS > 0.30, energy is elevated.
If elevated during cutting: check tool condition and cutting parameters.
If elevated when machine is stopped: check auxiliary equipment (coolant pump, conveyor).
Correlate with kurtosis — loud + high kurtosis = mechanical fault.
Loud + low kurtosis + normal MTConnect = heavy cutting load, may be normal for that operation.

Q: What does a bearing fault sound like?
A: Early bearing fault: subtle increase in kurtosis (3.5-5), slight spectral centroid rise.
Advanced bearing fault: kurtosis > 10, high crest factor > 5, harmonic ratio drops below 0.4,
broadband noise replaces the clean tonal spindle sound.
The machine may not alarm until the bearing is severely damaged — acoustic monitoring
catches it 2-4 weeks earlier than load monitoring alone.

Q: How do I interpret the pitch data?
A: Pitch mean should be stable and correlate with spindle RPM.
Pitch std > 30 Hz means the machine speed is fluctuating — possible load variation,
drive issue, or intermittent mechanical resistance.
A sudden pitch spike (max >> mean) indicates a momentary overspeed or resonance event.
Compare pitch mean with (MS1speed / 60) * fundamental harmonic number to verify correlation.

# N64 / VR4300 opcode formats (quick reference)

Source: [en64 wiki — Opcodes](https://en64.shoutwiki.com/wiki/Opcodes).  
This document reformats the wiki’s “quick opcode reference sheet” into tables. It excludes **exceptions**, **COP0**, and **pseudo-instructions**, matching the original brief list.

---

## R format (`SPECIAL`, `funct` field)

| Instruction | Operands | Description |
|-------------|----------|-------------|
| ADD | rd, rs, rt | ADD word |
| ADDU | rd, rs, rt | Add Unsigned word |
| AND | rd, rs, rt | AND |
| DADD | rd, rs, rt | Doubleword ADD |
| DADDU | rd, rs, rt | Doubleword ADD Unsigned |
| DDIV | rs, rt | Doubleword DIVide |
| DDIVU | rs, rt | Doubleword DIVide Unsigned |
| DIV | rs, rt | DIVide word |
| DIVU | rs, rt | DIVide Unsigned word |
| DMULT | rs, rt | Doubleword MULTiply |
| DMULTU | rs, rt | Doubleword MULTiply Unsigned |
| DSLL | rd, rt, sa | Doubleword Shift Left Logical |
| DSLL32 | rd, rt, sa | Doubleword Shift Left Logical +32 |
| DSLLV | rd, rt, rs | Doubleword Shift Left Logical Variable |
| DSRA | rd, rt, sa | Doubleword Shift Right Arithmetic |
| DSRA32 | rd, rt, sa | Doubleword Shift Right Arithmetic +32 |
| DSRAV | rd, rt, rs | Doubleword Shift Right Arithmetic Variable |
| DSRL | rd, rt, sa | Doubleword Shift Right Logical |
| DSRL32 | rd, rt, sa | Doubleword Shift Right Logical +32 |
| DSRLV | rd, rt, rs | Doubleword Shift Right Logical Variable |
| DSUB | rd, rs, rt | Doubleword SUBtract |
| DSUBU | rd, rs, rt | Doubleword SUBtract Unsigned |
| MFHI | rd | Move From HI register |
| MFLO | rd | Move From LO register |
| MTHI | rd | Move To HI register |
| MTLO | rd | Move To LO register |
| MULT | rs, rt | MULTiply word |
| MULTU | rs, rt | MULTiply Unsigned word |
| NOR | rd, rs, rt | Not OR |
| OR | rd, rs, rt | OR |
| SLL | rd, rt, sa | Shift word Left Logical |
| SLLV | rd, rt, rs | Shift word Left Logical Variable |
| SLT | rd, rs, rt | Set on Less Than |
| SLTU | rd, rs, rt | Set on Less Than Unsigned |
| SRA | rd, rt, sa | Shift word Right Arithmetic |
| SRAV | rd, rt, rs | Shift word Right Arithmetic Variable |
| SRL | rd, rt, sa | Shift word Right Logical |
| SRLV | rd, rt, rs | Shift word Right Logical Variable |
| SUB | rd, rs, rt | SUBtract word |
| SUBU | rd, rs, rt | SUBtract Unsigned word |
| XOR | rd, rs, rt | eXclusive OR |

---

## I format

### Arithmetic / logical immediate

| Instruction | Operands | Description |
|-------------|----------|-------------|
| ADDI | rt, rs, imm | ADD Immediate word |
| ADDIU | rt, rs, imm | Add Immediate Unsigned word |
| ANDI | rt, rs, imm | AND Immediate |
| DADDI | rt, rs, imm | Doubleword ADD Immediate |
| DADDIU | rt, rs, imm | Doubleword ADD Immediate Unsigned |
| LUI | rt, imm | Load Upper Immediate |
| ORI | rt, rs, imm | OR Immediate |
| SLTI | rt, rs, imm | Set on Less Than Immediate |
| SLTIU | rt, rs, imm | Set on Less Than Immediate Unsigned |

### Loads / stores / memory

| Instruction | Operands | Description |
|-------------|----------|-------------|
| LB | rt, offset(base) | Load Byte |
| LBU | rt, offset(base) | Load Byte Unsigned |
| LD | rt, offset(base) | Load Doubleword |
| LDL | rt, offset(base) | Load Doubleword Left |
| LDR | rt, offset(base) | Load Doubleword Right |
| LH | rt, offset(base) | Load Halfword |
| LHU | rt, offset(base) | Load Halfword Unsigned |
| LL | rt, offset(base) | Load Linked word |
| LLD | rt, offset(base) | Load Linked Doubleword |
| LW | rt, offset(base) | Load Word |
| LWL | rt, offset(base) | Load Word Left |
| LWR | rt, offset(base) | Load Word Right |
| LWU | rt, offset(base) | Load Word Unsigned |
| SB | rt, offset(base) | Store Byte |
| SC | rt, offset(base) | Store Conditional word |
| SCD | rt, offset(base) | Store Conditional Doubleword |
| SD | rt, offset(base) | Store Doubleword |
| SDL | rt, offset(base) | Store Doubleword Left |
| SDR | rt, offset(base) | Store Doubleword Right |
| SH | rt, offset(base) | Store Halfword |
| SW | rt, offset(base) | Store Word |
| SWL | rt, offset(base) | Store Word Left |
| SWR | rt, offset(base) | Store Word Right |
| SYNC | | SYNChronize shared memory |

*Load delay slot* applies to load/use patterns on real hardware (emulators often schedule explicitly).

---

## J format (branches and jumps grouped)

Some encodings are structurally I-type; they are grouped here with branches/jumps.

| Instruction | Operands | Description |
|-------------|----------|-------------|
| BEQ | rs, rt, offset | Branch on = |
| BEQL | rs, rt, offset | Branch on EQual Likely |
| BGEZ | rs, offset | Branch on >= Zero |
| BGEZAL | rs, offset | Branch on >= Zero And Link |
| BGEZALL | rs, offset | Branch on >= Zero And Link Likely |
| BGEZL | rs, offset | Branch on >= Equal to Zero Likely |
| BGTZ | rs, offset | Branch on > Zero |
| BGTZL | rs, offset | Branch on > Zero Likely |
| BLEZ | rs, offset | Branch on <= Equal to Zero |
| BLEZL | rs, offset | Branch on <= Zero Likely |
| BLTZ | rs, offset | Branch on < Zero |
| BLTZAL | rs, offset | Branch on < Zero And Link |
| BLTZALL | rs, offset | Branch on < Zero And Link Likely |
| BLTZL | rs, offset | Branch on < Zero Likely |
| BNE | rs, rt, offset | Branch on <> |
| BNEL | rs, rt, offset | Branch on <> Likely |
| J | target | Jump |
| JAL | target | Jump And Link |
| JALR | rs, rd | Jump And Link Register |
| JR | rs | Jump Register |

*Jump / branch delay slot* applies to these instructions.

---

## Floating-point formats

Function fields are not fully enumerated here; grouping follows the original wiki.

### Float R-type

| Instruction | Operands | Description |
|-------------|----------|-------------|
| ABS.fmt | fd, fs | floating-point Absolute value |
| ADD.fmt | fd, fs, ft | floating-point ADD |
| C.cond.fmt | fs, ft | floating-point Compare |
| CEIL.L.fmt | fd, fs | CEILing convert to Long fixed-point |
| CEIL.W.fmt | fd, fs | CEILing convert to Word fixed-point |
| CFC1 | rt, fs | Move control word From Floating-Point |
| CTC1 | rt, fs | Move control word To Floating-Point |
| CVT.D.fmt | fd, fs | ConVerT to Double floating-point |
| CVT.L.fmt | fd, fs | ConVerT to Long fixed-point |
| CVT.S.fmt | fd, fs | ConVerT to Single floating-point |
| CVT.W.fmt | fd, fs | ConVerT to Word fixed-point |
| DIV.fmt | fd, fs, ft | floating-point DIVide |
| DMFC1 | rt, fs | Doubleword Move From Floating-Point |
| DMTC1 | rt, fs | Doubleword Move To Floating-Point |
| FLOOR.L.fmt | fd, fs | FLOOR convert to Long fixed-point |
| FLOOR.W.fmt | fd, fs | FLOOR convert to Word fixed-point |
| MFC1 | rt, fs | Move Word From Floating-Point |
| MOV.fmt | fd, fs | floating-point MOVe |
| MTC1 | rt, fs | Move Word To Floating-Point |
| MUL.fmt | fd, fs, ft | floating-point MULtiply |
| NEG.fmt | fd, fs | floating-point NEGate |
| ROUND.L.fmt | fd, fs | ROUND to Long fixed-point |
| ROUND.W.fmt | fd, fs | ROUND to Word fixed-point |
| SQRT.fmt | fd, fs | SQuare RooT |
| SUB.fmt | fd, fs, ft | floating-point SUBtract |
| TRUNC.L.fmt | fd, fs | TRUNCate to Long fixed-point |
| TRUNC.W.fmt | fd, fs | TRUNCate to Word fixed-point |

### Float I-type

| Instruction | Operands | Description |
|-------------|----------|-------------|
| LDC1 | ft, offset(base) | Load Doubleword to Floating-Point |
| LWC1 | ft, offset(base) | Load Word to Floating-Point |
| SDC1 | ft, offset(base) | Store Doubleword from Floating-Point |
| SWC1 | ft, offset(base) | Store Word from Floating-Point |

### Float B-type

| Instruction | Operands | Description |
|-------------|----------|-------------|
| BC1F | offset | Branch on FP False |
| BC1FL | offset | Branch on FP False Likely |
| BC1T | offset | Branch on FP True |
| BC1TL | offset | Branch on FP True Likely |

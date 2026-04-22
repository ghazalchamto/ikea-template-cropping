Build a machine-readable ground-truth template specification repository from the uploaded PDF. Do not produce a raw extraction dump.

Your job is to convert the PDF into a normalized JSON spec system for downstream template validation.

Requirements:

1. Create a complete template inventory of all template variants in the PDF.
   - Output: review/templates_index.json
   - Include template id, display name, page number, family, x-dimension if applicable, notes, confidence, needs_review.

2. Extract shared/global barcode and DataMatrix specs into centralized files.
   - Output: shared_specs/barcode_specs.json
   - Reuse these by reference from templates.
   - Include DataMatrix rectangular/square specs, ITF specs, EAN specs when present.

3. Define JSON schemas before extraction.
   - Output:
     - schema/template.schema.json
     - schema/element.schema.json
     - schema/rule.schema.json
     - schema/shared_spec.schema.json

4. Create family/base definitions for related templates.
   - Output: families/*.json
   - Put shared elements/rules/layout assumptions there.

5. Create one JSON file per template.
   - Output: templates/*.json
   - Include identity, inheritance, elements, rules, overrides, evidence, review state.

6. For each template, extract all meaningful elements.
   - Examples: product name, article number, IKEA logo, country text, address block, DataMatrix block, ITF block, legal symbols, patents block, compliance text, date block, dimensions, weight, colour dot, main product illustration, vertical strip, copy count.
   - For each element, store:
     - element_id
     - label
     - semantic_role
     - type
     - required
     - repeatable
     - content_type
     - example_value
     - allowed_patterns
     - placement
     - orientation
     - shared_spec_ref if applicable
     - source evidence

7. Convert every note/condition/instruction into a structured rule object.
   - Use rule types such as:
     - presence_rule
     - conditional_layout_transform
     - alignment_rule
     - placement_rule
     - resize_rule
     - orientation_rule
     - application_method_rule
     - symbol_logic_rule
   - Every rule must include:
     - rule_id
     - rule_type
     - scope
     - condition
     - target
     - actions
     - severity
     - normalized_text
     - source evidence

8. Use inheritance and overrides.
   - Do not duplicate related templates unnecessarily.
   - Keep shared behavior at family level and local differences in overrides.

9. Attach PDF evidence to every important fact.
   - At minimum include page number and evidence type.
   - Add snippet text where practical.

10. Record ambiguity explicitly.
   - Output: review/open_questions.json
   - Do not silently guess uncertain interpretations.

11. Provide a validation script.
   - Output: validation/validate_specs.py
   - Must validate schemas, references, duplicate ids, missing evidence, and inventory/template consistency.

12. Repository structure must be:

ground_truth/
  source_pdf/
  schema/
  shared_specs/
  families/
  templates/
  review/
  validation/

Important rules:
- Do not treat placeholders/examples as literal fixed values unless clearly stated.
- Do not repeat shared barcode specs inside every template.
- Do not leave behavioral notes as plain notes; turn them into structured rules.
- Prefer semantic zones and relative placement over brittle exact coordinates unless exact geometry is reliable.
- Keep everything traceable to the PDF.

Definition of done:
- full inventory exists
- shared specs extracted
- schemas defined
- all templates extracted
- all important rules normalized
- inheritance used
- evidence attached
- ambiguities documented
- validation script passes

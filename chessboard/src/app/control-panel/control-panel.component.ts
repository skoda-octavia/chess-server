import { Component, OnInit, Output, EventEmitter } from '@angular/core';

@Component({
  selector: 'app-control-panel',
  templateUrl: './control-panel.component.html',
  styleUrls: ['./control-panel.component.css']
})
export class ControlPanelComponent {
  @Output() resetClicked = new EventEmitter<void>();
  @Output() reverseClicked = new EventEmitter<void>();
  @Output() optionChanged = new EventEmitter<string>();

  onOptionChange(event: Event): void {
    const selectElement = event.target as HTMLSelectElement;
    this.optionChanged.emit(selectElement.value);
  }

  Reset() {
    this.resetClicked.emit();
  }

  Reverse() {
    this.reverseClicked.emit();
  }
}
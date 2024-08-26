import { Component, OnInit, ViewChild, Output } from '@angular/core';
import {NgxChessBoardService} from 'ngx-chess-board';
import {NgxChessBoardView} from 'ngx-chess-board';
import { ApiService } from './api.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  @ViewChild('board', { static: false })
  board!: NgxChessBoardView;

  turn = true;
  isGameFinished = false;
  endgameMessage = '';
  selectedOption = "LSTM"
  isProcessingMove = false;

  constructor(
    private ngxChessBoardService: NgxChessBoardService,
    private apiService: ApiService) {}

  ngOnInit(): void {}

  ngAfterViewInit() {
    this.board.reset();
  }

  switchPlayerTurn() {
    this.turn = !this.turn;
  }

  MoveCompleted(event: any) {
    console.log(event)

    if (event.checkmate && this.turn) {
      this.endgameMessage = 'white won!';
      this.isGameFinished = true;
      console.warn(this.endgameMessage)
      return;
    }

    if (event.stalement) {
      this.endgameMessage = 'draw!';
      this.isGameFinished = true;
      console.warn(this.endgameMessage)
      return;
    }

    if (this.isProcessingMove) {
      return;
    }
    this.isProcessingMove = true;
    let fen = this.board.getFEN()
    console.log(fen);
    this.apiService.getData('get-move', this.selectedOption, fen).subscribe(
      response => {
        console.log('Data:', response);
        this.board.move(response.move);
        this.isProcessingMove = false;
      },
      error => {
        console.error('Error:', error);
        this.isProcessingMove = false;
      }
    );
    this.switchPlayerTurn();
  }

  onResetGame() {
    this.board.reset();
    this.turn = true;
    this.endgameMessage = '';
    this.isGameFinished = false;
  }

  Reset() {
    this.board.reset();
    this.isGameFinished = false;
  }

  Reverse() {
    this.board.reverse()
  }

  handleOptionChange(option: string): void {
    this.selectedOption = option;
    console.log('Selected option:', this.selectedOption);
  }
}
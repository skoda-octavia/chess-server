import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { NgxChessBoardModule } from "ngx-chess-board";
import { ToolbarComponent } from './toolbar/toolbar.component';
import { ControlPanelComponent } from './control-panel/control-panel.component';
import { HttpClientModule } from '@angular/common/http'; // Importowanie HttpClientModule
import { ApiService } from './api.service'; // Importowanie serwisu

@NgModule({
  declarations: [
    AppComponent,
    ToolbarComponent,
    ControlPanelComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    NgxChessBoardModule,
    HttpClientModule
  ],
  providers: [ApiService],
  bootstrap: [AppComponent]
})
export class AppModule { }
